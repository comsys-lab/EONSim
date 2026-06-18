import numpy as np
import argparse

## We implement this module based on this code: https://github.com/rishucoding/reproduce_MICRO24_GPU_DLRM_inference by RJ

## Assisting the args parser
def dash_separated_ints(value):
    vals = value.split("-")
    for val in vals:
        try:
            int(val)
        except ValueError:
            raise argparse.ArgumentTypeError(
                "%s is not a valid dash separated list of ints" % value
            )

    return value

class AddrGenerator:
    """Computes embedding-access byte addresses from the vector-index trace (lS_i), so the
    full address trace does not need to be stored; only lS_i is kept. Supports
    len()/indexing/iteration and can be indexed as emb_dataset[nb][nt]. A one-batch cache
    covers repeated access within a batch (batches are processed in order).
    """

    def __init__(self, lS_i, vector_stride, mem_gran, access_per_vector, rows_per_table):
        self.lS_i = lS_i
        self.vector_stride = int(vector_stride)
        self.mem_gran = int(mem_gran)
        # Number of cache-line-sized accesses per embedding vector.
        self.apv = int(access_per_vector)
        self.rows_per_table = int(rows_per_table)
        self.nbatches = len(lS_i)
        self.num_tables = len(lS_i[0]) if self.nbatches else 0
        self._dim_offsets = np.arange(self.apv, dtype=np.int64) * self.mem_gran
        self._cache_nb = -1
        self._cache_tables = None

    def table_addrs(self, nb, nt):
        """Byte addresses (1D int64) for one batch/table."""
        idx = np.asarray(self.lS_i[nb][nt], dtype=np.int64)
        table_base = nt * self.rows_per_table * self.vector_stride
        addr = table_base + idx[:, None] * self.vector_stride + self._dim_offsets[None, :]
        return addr.reshape(-1)

    def _batch_tables(self, nb):
        if nb != self._cache_nb:
            self._cache_tables = [self.table_addrs(nb, nt) for nt in range(self.num_tables)]
            self._cache_nb = nb
        return self._cache_tables

    def flat_addrs(self):
        """The whole trace flattened in batch -> table -> vector order (used by OPT)."""
        return np.concatenate([self.table_addrs(nb, nt)
                               for nb in range(self.nbatches)
                               for nt in range(self.num_tables)])

    # Read-only indexing as emb_dataset[nb][nt].
    def __len__(self):
        return self.nbatches

    def __getitem__(self, nb):
        return _BatchAddrView(self, nb)

    def __iter__(self):
        for nb in range(self.nbatches):
            yield _BatchAddrView(self, nb)


class _BatchAddrView:
    """View of one batch's per-table address arrays. len() returns the table count without
    generating; indexing or iterating triggers generation (cached by AddrGenerator)."""

    __slots__ = ("_gen", "_nb")

    def __init__(self, gen, nb):
        self._gen = gen
        self._nb = nb

    def __len__(self):
        return self._gen.num_tables

    def __getitem__(self, nt):
        tables = self._gen._batch_tables(self._nb)
        return tables[nt]  # int index -> one ndarray; slice -> list of ndarrays

    def __iter__(self):
        return iter(self._gen._batch_tables(self._nb))


class ReqGenerator:
    def __init__(self, nbatches, n_format_byte, embsize, emb_dim, bsz, fname, num_indices_per_lookup, mem_gran, debug=False):
        self.dataset_gen = None
        self.lS_o = []
        self.lS_i = []
        self.addr_trace = None   # addresses come from addr_gen (set in index_to_addr)
        self.addr_gen = None
        self.debug = debug

        self.nbatches = nbatches
        self.n_format_byte = n_format_byte
        self.embsize = embsize
        self.emb_dim = emb_dim
        self.bsz = bsz
        self.fname = fname
        self.num_indices_per_lookup = num_indices_per_lookup
        self.mem_gran = mem_gran

        self.access_per_vector = np.ceil(self.emb_dim * self.n_format_byte / self.mem_gran).astype(np.int32)

    def _load_filtered_indices(self, rows):
        # Read the whole dataset once and keep only indices < rows (= rows_per_table).
        # np.array parses in C, which is faster than a Python int() loop.
        with open(self.fname) as f:
            raw = np.array(f.read().split(), dtype=np.int64)
        return raw[raw < np.int64(rows)]

    def data_gen(self):
        # Build the index trace by reading the filtered index stream in order
        #     batch -> table -> sample -> k (k in range(num_indices_per_lookup)),
        # i.e. global position p = (((j*num_tables + t)*bsz) + s)*num_indices_per_lookup + k,
        # value filtered_idx[p % len]. np.take(..., mode="wrap") does this in one call.
        # Processing one batch at a time keeps the temporary index array to a single batch.
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)

        filtered = self._load_filtered_indices(int(ln_emb[0]))
        num_tables = int(len(ln_emb))
        K = int(self.num_indices_per_lookup)
        per_table = self.bsz * K
        per_batch = num_tables * per_table

        # Offsets are the same for every (batch, table): [0, K, 2K, ..., (bsz-1)*K].
        offsets_template = np.arange(self.bsz, dtype=np.int64) * K

        for j in range(self.nbatches):
            start = j * per_batch
            flat = np.take(
                filtered,
                np.arange(start, start + per_batch, dtype=np.int64),
                mode="wrap",
            )
            batch_indices = flat.reshape(num_tables, per_table)
            # Rows of a C-contiguous array are contiguous int64 views (safe to pickle).
            self.lS_i.append([batch_indices[t] for t in range(num_tables)])
            self.lS_o.append([offsets_template for _ in range(num_tables)])

    def index_to_addr(self):
        # Build an AddrGenerator that computes addresses from self.lS_i. The on-chip model
        # and policies index it as emb_dataset[nb][nt].
        ln_emb = np.fromstring(self.embsize, dtype=int, sep="-")
        ln_emb = np.asarray(ln_emb, dtype=np.int32)
        rows_per_table = int(ln_emb[0])

        if self.debug:
            print("[DEBUG] lS_i shape: {}".format(np.array(self.lS_i).shape))

        vector_bytes = self.emb_dim * self.n_format_byte
        vector_stride = ((vector_bytes + self.mem_gran - 1) // self.mem_gran) * self.mem_gran

        self.addr_gen = AddrGenerator(
            self.lS_i,
            vector_stride=vector_stride,
            mem_gran=self.mem_gran,
            access_per_vector=int(self.access_per_vector),
            rows_per_table=rows_per_table,
        )
        self.addr_trace = self.addr_gen   # the simulator reads this as emb_dataset

    def get_unique_vector_size_stats(self):
        # Quick workload-footprint diagnostics from vector-index trace.
        if len(self.lS_i) == 0:
            return {
                "vector_bytes": self.emb_dim * self.n_format_byte,
                "per_batch": [],
                "all_batches_unique_vectors": 0,
                "all_batches_unique_bytes": 0,
            }

        num_tables = len(self.lS_i[0])
        vector_bytes = self.emb_dim * self.n_format_byte
        global_unique_per_table = [set() for _ in range(num_tables)]
        per_batch_stats = []

        for batch_idx, batch_trace in enumerate(self.lS_i):
            batch_unique_vectors = 0

            for table_id in range(num_tables):
                table_indices = batch_trace[table_id]
                unique_vec = np.unique(table_indices)
                batch_unique_vectors += int(unique_vec.size)
                global_unique_per_table[table_id].update(map(int, unique_vec))

            per_batch_stats.append({
                "batch_idx": batch_idx,
                "unique_vectors": batch_unique_vectors,
                "unique_bytes": batch_unique_vectors * vector_bytes,
            })

        all_batches_unique_vectors = sum(len(s) for s in global_unique_per_table)

        return {
            "vector_bytes": vector_bytes,
            "per_batch": per_batch_stats,
            "all_batches_unique_vectors": all_batches_unique_vectors,
            "all_batches_unique_bytes": all_batches_unique_vectors * vector_bytes,
        }

    def do_batch_access_pattern_analysis(self):
        # Convert first batch (batch 0) addresses into a list to maintain duplicates
        first_batch_addrs = []
        for table in self.addr_trace[0]:
            first_batch_addrs.extend(table)

        total_addrs = len(first_batch_addrs)  # Total number of addresses including duplicates
        first_batch_set = set(first_batch_addrs)  # Set for intersection

        print("\nBatch Access Pattern Analysis:")
        print("--------------------------------")
        print(f"Total addresses in each batch: {total_addrs}")
        print("Overlap percentages with first batch (Batch 0):")

        # Compare with all batches (including batch 0)
        for batch_idx in range(0, len(self.addr_trace)):
            current_batch_addrs = []
            for table in self.addr_trace[batch_idx]:
                current_batch_addrs.extend(table)

            # Calculate overlap using sets for unique addresses
            current_batch_set = set(current_batch_addrs)
            overlap_addrs = first_batch_set.intersection(current_batch_set)

            # Calculate percentage based on total addresses (including duplicates)
            overlap_count = sum(1 for addr in first_batch_addrs if addr in current_batch_set)
            overlap_percentage = (overlap_count / total_addrs) * 100

            print(f"Batch {batch_idx}: {overlap_percentage:.2f}%")

if __name__ == "__main__":
    raise SystemExit("ReqGenerator is an internal module and is not intended for standalone execution.")
