from collections import Counter
import numpy as np
import heapq
from cache_modules.LRU_module import LRU_module
from cache_modules.LFU_module import LFU_module
from cache_modules.SRRIP_module import SRRIP_module
import itertools
from tqdm import tqdm
import random

class OnmemPolicy:
    def __init__(self, cache_config):
        self.cache_way = cache_config['way']
        self.cache_set = cache_config['set_count']

    def initialize(self):
        pass

    def handle_access(self, tag, index):
        raise NotImplementedError

    def begin_batch(self, batch_idx):
        pass

    def end_batch(self, batch_idx):
        pass

    def post_access_processing(self, hit, tag, index, vec):
        pass

class LRUPolicy(OnmemPolicy):
    def initialize(self):
        self.on_mem = [LRU_module(self.cache_way) for _ in range(self.cache_set)]

    def handle_access(self, tag, index):
        if self.on_mem[index].search_and_access(tag):
            return True, None
        self.on_mem[index].insert_node(tag)
        return False, None

class SRRIPPolicy(OnmemPolicy):
    def __init__(self, cache_config):
        super().__init__(cache_config)
        self.rrpv_bits = cache_config['rrpv_bits']
        self.rrpv_insert = cache_config['rrpv_insert']

    def initialize(self):
        self.on_mem = [SRRIP_module(self.cache_way, self.rrpv_bits, self.rrpv_insert) for _ in range(self.cache_set)]

    def handle_access(self, tag, index):
        return self.on_mem[index].access(tag), None


class LFUPolicy(OnmemPolicy):
    def __init__(self, cache_config):
        super().__init__(cache_config)
        self.counter_bits = cache_config.get('lfu_counter_bits', 8)
        self.aging_interval = cache_config.get('lfu_aging_interval', 0)

    def initialize(self):
        self.on_mem = [
            LFU_module(self.cache_way, self.counter_bits, self.aging_interval)
            for _ in range(self.cache_set)
        ]

    def handle_access(self, tag, index):
        return self.on_mem[index].access(tag)

class OptPolicy(OnmemPolicy):
    def __init__(self, cache_config, emb_dataset, num_cores=1, enable_bypass=False):
        super().__init__(cache_config)
        
        self.cache_way = cache_config['way']
        self.cache_line_size = cache_config['line_size']
        self.cache_set = cache_config['set_count']
        self.num_cores = num_cores
        self.enable_bypass = bool(enable_bypass)
        
        cache_offset_bits = int(np.log2(self.cache_line_size-1)+1)
        offset_mask_bits = (1 << cache_offset_bits) - 1
        
        print("Preprocessing OPT trace (Flattening & Line Identification)...")
        
        if num_cores > 1:
            # Multicore: flatten in round-robin order
            flat_addr = self._flatten_multicore_roundrobin(emb_dataset)
        else:
            # Single-core: flatten in batch->table order
            all_arrays = [t for batch in emb_dataset for t in batch]
            flat_addr = np.concatenate(all_arrays)
        
        self.flat_lines = flat_addr & ~offset_mask_bits
        
        print("Preprocessing OPT trace (Calculating Next Use)...")
        trace_len = len(self.flat_lines)
        self.next_access = np.full(trace_len, trace_len + 1, dtype=np.int64)
        
        last_seen = {}
        for i in range(trace_len - 1, -1, -1):
            line = self.flat_lines[i]
            if line in last_seen:
                self.next_access[i] = last_seen[line]
            last_seen[line] = i
            
        self.curr_cycle = 0
        print("OPT Preprocessing Done.")
    
    def _flatten_multicore_roundrobin(self, emb_dataset):
        """Flatten dataset in batch-barriered round-robin order matching runtime simulation."""
        # Partition tables across cores
        num_tables = len(emb_dataset[0])
        tables_per_core = num_tables // self.num_cores
        remainder = num_tables % self.num_cores
        
        table_ranges = []
        start_idx = 0
        for core_id in range(self.num_cores):
            extra = 1 if core_id < remainder else 0
            end_idx = start_idx + tables_per_core + extra
            table_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        
        # Create iterators for each core
        class SimpleIterator:
            def __init__(self, dataset, table_start, table_end):
                self.dataset = dataset
                self.table_start = table_start
                self.table_end = table_end
                self.batch_idx = 0
                self.table_idx = table_start
                self.vec_idx = 0
                self.total_accesses = self._count_total_accesses()
                self.current_access = 0

            def _count_total_accesses(self):
                total = 0
                for batch in self.dataset:
                    for table_id in range(self.table_start, self.table_end):
                        total += len(batch[table_id])
                return total
            
            def has_next(self):
                return self.current_access < self.total_accesses
            
            def get_next(self):
                if not self.has_next():
                    return None
                addr = self.dataset[self.batch_idx][self.table_idx][self.vec_idx]
                self._advance()
                return addr
            
            def _advance(self):
                self.vec_idx += 1
                self.current_access += 1
                if self.vec_idx >= len(self.dataset[self.batch_idx][self.table_idx]):
                    self.vec_idx = 0
                    self.table_idx += 1
                    if self.table_idx >= self.table_end:
                        self.table_idx = self.table_start
                        self.batch_idx += 1
                        if self.batch_idx >= len(self.dataset):
                            self.batch_idx = len(self.dataset) - 1
        
        iterators = [
            (table_start, table_end)
            for table_start, table_end in table_ranges
        ]

        # Round-robin collection with a batch barrier.
        # This keeps access order aligned with runtime cache simulation,
        # where all cores finish one batch before the next batch starts.
        flat_list = []
        for nb in range(len(emb_dataset)):
            single_batch = [emb_dataset[nb]]
            batch_iterators = [
                SimpleIterator(single_batch, table_start, table_end)
                for table_start, table_end in iterators
            ]

            # Iterate only active cores to avoid sparse-core tick overhead.
            active_core_ids = [core_id for core_id, it in enumerate(batch_iterators) if it.has_next()]
            rr_idx = 0

            while active_core_ids:
                core_id = active_core_ids[rr_idx]
                iterator = batch_iterators[core_id]

                addr = iterator.get_next()
                if addr is not None:
                    flat_list.append(addr)

                if iterator.has_next():
                    rr_idx = (rr_idx + 1) % len(active_core_ids)
                else:
                    del active_core_ids[rr_idx]
                    if active_core_ids:
                        rr_idx %= len(active_core_ids)
        
        return np.array(flat_list, dtype=np.int64)

    def initialize(self):
        # Fixed-size per-set state for faster hit lookup and victim selection.
        self.on_mem = np.zeros((self.cache_set, self.cache_way), dtype=np.int64)
        self.on_mem_next_use = np.full((self.cache_set, self.cache_way), -1, dtype=np.int64)
        self.valid_count = np.zeros(self.cache_set, dtype=np.int32)
        self.tag_to_way = [{} for _ in range(self.cache_set)]
        # Per-set max-heap over resident lines: key=(-next_use, way, version).
        # This preserves legacy tie-break: first maximum -> smallest way index.
        self.next_use_heaps = [[] for _ in range(self.cache_set)]
        self.way_versions = np.zeros((self.cache_set, self.cache_way), dtype=np.int64)

    def _push_way_state(self, set_idx, way):
        self.way_versions[set_idx, way] += 1
        ver = int(self.way_versions[set_idx, way])
        next_use = int(self.on_mem_next_use[set_idx, way])
        heapq.heappush(self.next_use_heaps[set_idx], (-next_use, int(way), ver))

    def _peek_farthest_way(self, set_idx):
        heap = self.next_use_heaps[set_idx]
        while heap:
            neg_next, way, ver = heap[0]
            if ver != int(self.way_versions[set_idx, way]):
                heapq.heappop(heap)
                continue
            if way >= int(self.valid_count[set_idx]):
                heapq.heappop(heap)
                continue
            return int(way), int(-neg_next)
        return None, None

    def _pop_farthest_way(self, set_idx):
        heap = self.next_use_heaps[set_idx]
        while heap:
            neg_next, way, ver = heapq.heappop(heap)
            if ver != int(self.way_versions[set_idx, way]):
                continue
            if way >= int(self.valid_count[set_idx]):
                continue
            return int(way), int(-neg_next)
        return None, None

    def handle_access(self, tag, index, **kwargs):
        # Get the precomputed next use index for the current access
        # Since self.curr_cycle corresponds to the exact sequence of access,
        # self.next_access[self.curr_cycle] correctly points to the next time *this specific line* is used.
        current_next_use = int(self.next_access[self.curr_cycle])
        tag = int(tag)

        set_map = self.tag_to_way[index]
        way = set_map.get(tag)

        if way is not None:
            self.on_mem_next_use[index, way] = current_next_use
            self._push_way_state(index, way)
            return True, None

        valid = int(self.valid_count[index])
        if valid < self.cache_way:
            fill_way = valid
            self.on_mem[index, fill_way] = tag
            self.on_mem_next_use[index, fill_way] = current_next_use
            self.valid_count[index] = valid + 1
            set_map[tag] = fill_way
            self._push_way_state(index, fill_way)
            return False, None

        if self.enable_bypass:
            # Bypass when this miss has strictly farther next use than all resident lines.
            # This keeps off-chip miss accounting while avoiding cache pollution.
            _, farthest_resident_next_use = self._peek_farthest_way(index)
            if farthest_resident_next_use is None:
                raise RuntimeError("OPT heap state is empty for a full cache set")
            if current_next_use > farthest_resident_next_use:
                return False, None

        # Use first-maximum tie-break to match prior list.index(max(...)) behavior.
        victim_way, _ = self._pop_farthest_way(index)
        if victim_way is None:
            raise RuntimeError("OPT victim selection failed: no valid heap entry")
        victim_tag = int(self.on_mem[index, victim_way])

        del set_map[victim_tag]
        self.on_mem[index, victim_way] = tag
        self.on_mem_next_use[index, victim_way] = current_next_use
        set_map[tag] = victim_way
        self._push_way_state(index, victim_way)

        return False, None

    def post_access_processing(self, hit, tag, index, vec):
        self.curr_cycle += 1

class SpadPolicy(OnmemPolicy):
    def __init__(self, mem_size, mem_gran, emb_dim, n_format_byte, emb_dataset, vectors_per_table, prof_period, spad_policy, num_cores=1, debug=False):
        self.mem_size = mem_size
        self.mem_gran = mem_gran
        self.emb_dim = emb_dim
        self.n_format_byte = n_format_byte
        self.emb_dataset = emb_dataset
        self.vectors_per_table = vectors_per_table
        self.prof_period = prof_period
        self.spad_policy = spad_policy
        self.num_cores = num_cores
        self.debug = debug
        self.num_tables = len(self.emb_dataset[0])
        self.access_per_vector = np.ceil(self.emb_dim * self.n_format_byte / self.mem_gran).astype(np.int32)
        self.spad_size = np.floor(self.mem_size / self.mem_gran).astype(np.int32)
        self.batch_counter = 0

    def initialize(self):
        self.on_mem = self.set_spad()

    def handle_access(self, tag, index):
        # SPAD policies operate on the entire `on_mem` set, not on a per-index basis.
        # The "tag" here is the actual address.
        return tag in self.on_mem, None

    def set_spad(self):
        if self.spad_policy == "spad_naive":
            return self.set_spad_naive()
        elif self.spad_policy == "spad_random":
            return self.set_spad_random()
        elif self.spad_policy == "spad_oracle":
            return self.set_spad_oracle()
        else:
            raise NotImplementedError(f"SPAD policy {self.spad_policy} not implemented")

    def set_spad_naive(self):
        on_mem_set = []
        vector_bytes = self.emb_dim * self.n_format_byte
        vector_stride = ((vector_bytes + self.mem_gran - 1) // self.mem_gran) * self.mem_gran
        
        # Partition tables across cores (same logic as CoreOnmem._partition_tables_across_cores)
        tables_per_core = self.num_tables // self.num_cores
        remainder = self.num_tables % self.num_cores
        vectors_per_core = self.spad_size // self.num_cores
        
        with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
            counter = 0
            start_idx = 0
            
            for core_id in range(self.num_cores):
                # Calculate table range for this core (same as _partition_tables_across_cores)
                extra = 1 if core_id < remainder else 0
                table_end = start_idx + tables_per_core + extra
                
                if self.debug: print(f"[DEBUG] Core {core_id}: tables {start_idx}-{table_end-1}, allocating {vectors_per_core} memory accesses")
                
                # Allocate vectors_per_core for this core from its assigned tables
                core_counter = 0
                break_flag = False
                
                for t_i in range(start_idx, table_end):
                    table_base = t_i * self.vectors_per_table * vector_stride
                    for v_i in range(self.vectors_per_table):
                        row_base = v_i * vector_stride
                        for d_i in range(self.access_per_vector):
                            dim_offset = self.mem_gran * d_i
                            this_addr = table_base + row_base + dim_offset
                            on_mem_set.append(this_addr)
                            core_counter += 1
                            counter += 1
                            pbar.update(1)
                            
                            if core_counter >= vectors_per_core:
                                break_flag = True
                                break
                        if break_flag: break
                    if break_flag: break
                
                # Move to next core's table range
                start_idx = table_end
        
        if self.debug: print(f"[DEBUG] Total loaded to spad: {len(on_mem_set)} addresses")
        return set(on_mem_set)

    def set_spad_random(self):
        on_mem_set = []
        vector_bytes = self.emb_dim * self.n_format_byte
        vector_stride = ((vector_bytes + self.mem_gran - 1) // self.mem_gran) * self.mem_gran
        avail_space = list(itertools.product(range(self.num_tables), range(self.vectors_per_table)))
        random.shuffle(avail_space)
        avail_space = avail_space[:int(self.spad_size/self.access_per_vector)]
        with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
            for pair in avail_space:
                table_base = pair[0] * self.vectors_per_table * vector_stride
                row_base = pair[1] * vector_stride
                for d_i in range(self.access_per_vector):
                    dim_offset = self.mem_gran * d_i
                    this_addr = table_base + row_base + dim_offset
                    on_mem_set.append(this_addr)
                    pbar.update(1)
        return set(on_mem_set)

    def set_spad_oracle(self):
        end_batch = min(self.batch_counter + self.prof_period, len(self.emb_dataset))
        access_freq = Counter()
        for batch_idx in range(self.batch_counter, end_batch):
            if batch_idx >= len(self.emb_dataset): break
            for table in self.emb_dataset[batch_idx]:
                for addr in table.flatten():
                    access_freq[addr] += 1
        most_common = access_freq.most_common()
        top_accesses = most_common[:min(self.spad_size, len(most_common))]
        return set(x[0] for x in top_accesses)

    def post_access_processing(self, hit, tag, index, vec):
        if self.spad_policy == "spad_oracle":
            # This logic is handled in the main simulation loop of the driver
            pass


class ProfilePolicy(OnmemPolicy):
    def __init__(
        self,
        mem_size,
        mem_gran,
        emb_dim,
        n_format_byte,
        emb_dataset,
        vectors_per_table,
        prof_period,
        profile_policy,
        cache_config,
        index_trace=None,
        debug=False,
    ):
        super().__init__(cache_config)
        self.mem_size = mem_size * 1024  # KB -> B
        self.mem_gran = mem_gran
        self.emb_dim = emb_dim
        self.n_format_byte = n_format_byte
        self.emb_dataset = emb_dataset
        self.vectors_per_table = vectors_per_table
        self.prof_period = prof_period
        self.profile_policy = profile_policy
        self.index_trace = index_trace
        self.debug = debug

        self.num_tables = len(self.emb_dataset[0])
        self.access_per_vector = np.ceil(self.emb_dim * self.n_format_byte / self.mem_gran).astype(np.int32)
        self.spad_size = np.floor(self.mem_size / self.mem_gran).astype(np.int32)

        # Keep legacy behavior for profile SRRIP unless overridden by config.
        self.cache_way = cache_config.get('way', 0) or 128
        self.cache_line_size = cache_config.get('line_size', 0) or self.mem_gran
        self.cache_set = max(1, int(self.mem_size / self.cache_line_size / self.cache_way))

        self.rrpv_bits = cache_config.get('rrpv_bits', 4)
        self.rrpv_insert = cache_config.get('rrpv_insert', 14)

        self.access_results = []
        self.spad_load_results = []
        self.logger_results = []

        self._batch_hit = 0
        self._batch_miss = 0
        self._batch_spad_load = 0
        self._batch_logger_hit = 0
        self._batch_logger_miss = 0

    def initialize(self):
        if self.profile_policy == "profile_dynamic_cache":
            self.logger_size = self.spad_size
            self.logger = LRU_module(self.logger_size)
        elif self.profile_policy == "profile_dynamic_SRRIP":
            self.logger = [np.zeros((0, 2), dtype=np.int64) for _ in range(self.cache_set)]
        elif self.profile_policy == "profile_dynamic_count":
            if self.index_trace is None:
                raise ValueError("index_trace is required for profile_dynamic_count")
            self.counter_arr = np.zeros((1, len(self.index_trace[0]), self.vectors_per_table), dtype=np.int64)
            self.counter_set = 0
        else:
            raise NotImplementedError(f"Profile policy {self.profile_policy} not implemented")

        self.on_mem = self._set_spad()

    def begin_batch(self, batch_idx):
        self._batch_hit = 0
        self._batch_miss = 0
        self._batch_spad_load = 0
        self._batch_logger_hit = 0
        self._batch_logger_miss = 0

    def end_batch(self, batch_idx):
        self.access_results.append([self._batch_hit, self._batch_miss])
        self.spad_load_results.append(self._batch_spad_load)
        if self.profile_policy in {"profile_dynamic_cache", "profile_dynamic_SRRIP"}:
            self.logger_results.append([self._batch_logger_hit, self._batch_logger_miss])

    def refresh_on_mem(self):
        self.on_mem = self._set_spad()
        self._batch_spad_load += self.spad_size

    def _set_spad(self):
        on_mem_set = []
        vector_bytes = self.emb_dim * self.n_format_byte
        vector_stride = ((vector_bytes + self.mem_gran - 1) // self.mem_gran) * self.mem_gran

        if self.profile_policy in {"profile_dynamic_cache", "profile_dynamic_SRRIP"}:
            logger_empty = (
                self.profile_policy == "profile_dynamic_cache" and self.logger.is_empty()
            ) or (
                self.profile_policy == "profile_dynamic_SRRIP" and all(len(i) == 0 for i in self.logger)
            )

            if logger_empty:
                counter = 0
                break_flag = False
                for t_i in range(self.num_tables):
                    table_base = t_i * self.vectors_per_table * vector_stride
                    for v_i in range(self.vectors_per_table):
                        row_base = v_i * vector_stride
                        for d_i in range(self.access_per_vector):
                            dim_offset = self.mem_gran * d_i
                            this_addr = table_base + row_base + dim_offset
                            on_mem_set.append(this_addr)
                            counter += 1
                            if counter == self.spad_size:
                                break_flag = True
                                break
                        if break_flag:
                            break
                    if break_flag:
                        break
                on_mem_arr = np.asarray(on_mem_set, dtype=np.int64)
            else:
                if self.profile_policy == "profile_dynamic_cache":
                    on_mem_arr = self.logger.return_as_array()[:self.spad_size]
                else:
                    on_mem_arr = np.zeros(self.spad_size, dtype=np.int64)
                    for i in range(self.cache_set):
                        this_logger_len = len(self.logger[i])
                        if this_logger_len < self.cache_way:
                            on_mem_arr[i * self.cache_way:(i + 1) * self.cache_way] = np.pad(
                                self.logger[i][:this_logger_len, 0],
                                (0, self.cache_way - this_logger_len),
                                'constant',
                            )
                        else:
                            on_mem_arr[i * self.cache_way:(i + 1) * self.cache_way] = self.logger[i][:self.cache_way, 0]

        elif self.profile_policy == "profile_dynamic_count":
            if self.counter_set == 0:
                counter = 0
                break_flag = False
                for t_i in range(self.num_tables):
                    table_base = t_i * self.vectors_per_table * vector_stride
                    for v_i in range(self.vectors_per_table):
                        row_base = v_i * vector_stride
                        for d_i in range(self.access_per_vector):
                            dim_offset = self.mem_gran * d_i
                            this_addr = table_base + row_base + dim_offset
                            on_mem_set.append(this_addr)
                            counter += 1
                            if counter == self.spad_size:
                                break_flag = True
                                break
                        if break_flag:
                            break
                    if break_flag:
                        break
                self.counter_set = 1
                on_mem_arr = np.asarray(on_mem_set, dtype=np.int64)
            else:
                vectors_needed = self.spad_size // self.access_per_vector
                flat_indices = np.argpartition(self.counter_arr.ravel(), -vectors_needed)[-vectors_needed:]

                temp = flat_indices % (self.vectors_per_table * len(self.index_trace[0]))
                table_indices = temp // self.vectors_per_table
                vector_indices = temp % self.vectors_per_table

                dim_offsets = np.arange(self.access_per_vector, dtype=np.int64) * self.mem_gran
                table_base = table_indices[:, None] * (self.vectors_per_table * vector_stride)
                row_base = vector_indices[:, None] * vector_stride

                addresses = table_base + row_base + dim_offsets
                on_mem_arr = addresses.ravel()[:self.spad_size]
                self.counter_arr = np.zeros((1, len(self.index_trace[0]), self.vectors_per_table), dtype=np.int64)

        else:
            raise NotImplementedError(f"Profile policy {self.profile_policy} not implemented")

        self.on_mem_set = set(on_mem_arr)
        return on_mem_arr

    def _get_index_bits(self, addr):
        if self.cache_set <= 1:
            return 0
        cache_index_bits = int(np.ceil(np.log2(self.cache_set)))
        cache_offset_bits = int(np.log2(self.cache_line_size - 1) + 1)
        index_msb = cache_index_bits + cache_offset_bits - 1
        index_lsb = cache_offset_bits
        mask = ((1 << (index_msb - index_lsb + 1)) - 1) << index_lsb
        return ((addr & mask) >> index_lsb) % self.cache_set

    def handle_access(self, tag, index, **kwargs):
        addr = kwargs.get('addr', tag)
        table_id = kwargs.get('table_id', 0)
        vec_id = kwargs.get('vec_id', 0)
        batch_idx = kwargs.get('batch_idx', 0)

        is_hit = addr in self.on_mem_set
        if is_hit:
            self._batch_hit += 1
        else:
            self._batch_miss += 1

        if self.profile_policy == "profile_dynamic_cache":
            if not self.logger.search_and_access(addr):
                self.logger.insert_node(addr)
                self._batch_logger_miss += 1
            else:
                self._batch_logger_hit += 1

        elif self.profile_policy == "profile_dynamic_SRRIP":
            this_index = self._get_index_bits(addr)
            this_tag = addr
            tag_match = np.where(self.logger[this_index][:, 0] == this_tag)[0]

            if len(tag_match) > 0:
                self._batch_logger_hit += 1
                self.logger[this_index][tag_match[0], 1] = 0
            else:
                self._batch_logger_miss += 1
                if len(self.logger[this_index]) < self.cache_way:
                    new_entry = np.array([[this_tag, self.rrpv_insert]])
                    self.logger[this_index] = np.vstack([self.logger[this_index], new_entry])
                else:
                    max_rrpv = 2 ** self.rrpv_bits - 1
                    replaced = False
                    while not replaced:
                        victim_candidates = np.where(self.logger[this_index][:, 1] == max_rrpv)[0]
                        if len(victim_candidates) > 0:
                            self.logger[this_index][victim_candidates[0]] = [this_tag, self.rrpv_insert]
                            replaced = True
                        else:
                            self.logger[this_index][:, 1] = np.minimum(self.logger[this_index][:, 1] + 1, max_rrpv)

        elif self.profile_policy == "profile_dynamic_count":
            vec_ind = vec_id // self.access_per_vector
            this_vec_ind = self.index_trace[batch_idx][table_id][vec_ind]
            self.counter_arr[0][table_id][this_vec_ind] += 1

        return is_hit, None
