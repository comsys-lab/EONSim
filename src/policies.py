from collections import Counter
import numpy as np
from LRU_module import LRU_module
from SRRIP_module import SRRIP_module
import itertools
from collections import Counter
from tqdm import tqdm
import random

class CachePolicy:
    def __init__(self, cache_config):
        self.cache_way = cache_config[0]
        self.cache_set = cache_config[4]

    def initialize(self):
        pass

    def handle_access(self, tag, index):
        raise NotImplementedError

    def post_access_processing(self, hit, tag, index, vec):
        pass

class LRUPolicy(CachePolicy):
    def initialize(self):
        self.on_mem = [LRU_module(self.cache_way) for _ in range(self.cache_set)]

    def handle_access(self, tag, index):
        if self.on_mem[index].search_and_access(tag):
            return True, None  # Hit
        else:
            victim = self.on_mem[index].insert_node(tag)
            return False, victim  # Miss

class SRRIPPolicy(CachePolicy):
    def __init__(self, cache_config):
        super().__init__(cache_config)
        self.rrpv_bits = cache_config[2]
        self.rrpv_insert = cache_config[3]

    def initialize(self):
        self.on_mem = [SRRIP_module(self.cache_way, self.rrpv_bits, self.rrpv_insert) for _ in range(self.cache_set)]

    def handle_access(self, tag, index):
        return self.on_mem[index].access(tag), None

class OptPolicy(CachePolicy):
    def __init__(self, cache_config, emb_dataset):
        super().__init__(cache_config)
        
        # Reconstruct tag bit logic from cache_config
        # cache_config = [way, line_size, rrpv_bits, rrpv_insert, cache_set]
        self.cache_way = cache_config[0]
        self.cache_line_size = cache_config[1]
        self.cache_set = cache_config[4]
        
        # Calculate offset bits to identify unique cache lines (Block Address)
        cache_offset_bits = int(np.log2(self.cache_line_size-1)+1)
        offset_mask_bits = (1 << cache_offset_bits) - 1
        
        print("Preprocessing OPT trace (Flattening & Line Identification)...")
        # Flatten the dataset: Batch -> Table -> Access
        # emb_dataset is List[List[np.array]]
        all_arrays = [t for batch in emb_dataset for t in batch]
        flat_addr = np.concatenate(all_arrays)
        
        # Convert to Block Addresses (Tag + Index)
        # We must use the full block address to distinguish lines in different sets that share the same tag.
        self.flat_lines = flat_addr & ~offset_mask_bits
        
        print("Preprocessing OPT trace (Calculating Next Use)...")
        # Precompute next access indices (Belady's OPT)
        # next_access[i] contains the index j > i where the line at i is accessed next.
        trace_len = len(self.flat_lines)
        self.next_access = np.full(trace_len, trace_len + 1, dtype=np.int64)
        
        last_seen = {}
        # Iterate in reverse to find next usage
        for i in range(trace_len - 1, -1, -1):
            line = self.flat_lines[i]
            if line in last_seen:
                self.next_access[i] = last_seen[line]
            last_seen[line] = i
            
        self.curr_cycle = 0
        print("OPT Preprocessing Done.")

    def initialize(self):
        # on_mem stores the tags currently in cache
        self.on_mem = [[] for _ in range(self.cache_set)]
        # on_mem_next_use stores the 'next_access' index for each tag in on_mem
        self.on_mem_next_use = [[] for _ in range(self.cache_set)]

    def handle_access(self, tag, index):
        # Get the precomputed next use index for the current access
        # Since self.curr_cycle corresponds to the exact sequence of access,
        # self.next_access[self.curr_cycle] correctly points to the next time *this specific line* is used.
        current_next_use = self.next_access[self.curr_cycle]
        
        # Check for hit
        if tag in self.on_mem[index]:
            # Hit
            # Update the next_use for this tag
            pos = self.on_mem[index].index(tag)
            self.on_mem_next_use[index][pos] = current_next_use
            return True, None
        else:
            # Miss
            if len(self.on_mem[index]) < self.cache_way:
                # Fill invalid way
                self.on_mem[index].append(tag)
                self.on_mem_next_use[index].append(current_next_use)
                return False, None
            else:
                # Evict victim with furthest next use
                # Find index of max value in on_mem_next_use[index]
                victim_pos = self.on_mem_next_use[index].index(max(self.on_mem_next_use[index]))
                victim = self.on_mem[index][victim_pos]
                
                # Replace
                self.on_mem[index][victim_pos] = tag
                self.on_mem_next_use[index][victim_pos] = current_next_use
                
                return False, victim

    def post_access_processing(self, hit, tag, index, vec):
        self.curr_cycle += 1

class ProfilePolicy(LRUPolicy):
    def __init__(self, cache_config, emb_dataset, batch_counter):
        super().__init__(cache_config)
        self.emb_dataset = emb_dataset
        self.batch_counter = batch_counter
        self.profile_filter = self.create_profile_filter()

    def create_profile_filter(self):
        flat_dataset = itertools.chain.from_iterable(self.emb_dataset[self.batch_counter])
        access_freq = Counter(flat_dataset)
        access_freq = access_freq.most_common()
        new_access_freq = [(key, value) for key, value in access_freq if value >= 3]
        return np.array([x[0] for x in new_access_freq], dtype=np.int64)

    def handle_access(self, tag, index):
        if not tag in self.profile_filter:
            return False, None  # Miss due to profile filter
        return super().handle_access(tag, index)

    def post_access_processing(self, hit, tag, index, vec):
        self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset) - 1)
        self.profile_filter = self.create_profile_filter()

class SpadPolicy(CachePolicy):
    def __init__(self, mem_size, mem_gran, emb_dim, n_format_byte, emb_dataset, vectors_per_table, prof_multiplier, spad_policy):
        self.mem_size = mem_size
        self.mem_gran = mem_gran
        self.emb_dim = emb_dim
        self.n_format_byte = n_format_byte
        self.emb_dataset = emb_dataset
        self.vectors_per_table = vectors_per_table
        self.prof_multiplier = prof_multiplier
        self.spad_policy = spad_policy
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
        counter = 0
        break_flag = False
        with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
            for t_i in range(self.num_tables):
                for v_i in range(self.vectors_per_table):
                    for d_i in range(self.access_per_vector):
                        bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                        tbl_bits = t_i << int(np.log2(self.vectors_per_table-1)+1 + bytes_per_vec)
                        vec_idx = v_i << bytes_per_vec
                        dim_bits = self.mem_gran * d_i
                        this_addr = tbl_bits + vec_idx + dim_bits
                        on_mem_set.append(this_addr)
                        counter += 1
                        if counter == self.spad_size:
                            break_flag = True
                            break
                        pbar.update(1)
                    if break_flag: break
                if break_flag: break
        return set(on_mem_set)

    def set_spad_random(self):
        on_mem_set = []
        avail_space = list(itertools.product(range(self.num_tables), range(self.vectors_per_table)))
        random.shuffle(avail_space)
        avail_space = avail_space[:int(self.spad_size/self.access_per_vector)]
        with tqdm(total=self.spad_size, desc="Setting spad") as pbar:
            for pair in avail_space:
                for d_i in range(self.access_per_vector):
                    bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                    tbl_bits = pair[0] << int(np.log2(self.vectors_per_table) + bytes_per_vec)
                    vec_idx = pair[1] << bytes_per_vec
                    dim_bits = self.mem_gran * d_i
                    this_addr = tbl_bits + vec_idx + dim_bits
                    on_mem_set.append(this_addr)
                    pbar.update(1)
        return set(on_mem_set)

    def set_spad_oracle(self):
        end_batch = min(self.batch_counter + self.prof_multiplier, len(self.emb_dataset))
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
