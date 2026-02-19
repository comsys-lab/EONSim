from collections import Counter
import numpy as np
from cache_modules.LRU_module import LRU_module
from cache_modules.SRRIP_module import SRRIP_module
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
    def __init__(self, cache_config, emb_dataset, num_cores=1):
        super().__init__(cache_config)
        
        self.cache_way = cache_config[0]
        self.cache_line_size = cache_config[1]
        self.cache_set = cache_config[4]
        self.num_cores = num_cores
        
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
        """Flatten dataset in round-robin order matching multicore simulation"""
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
            
            def has_next(self):
                return self.batch_idx < len(self.dataset)
            
            def get_next(self):
                if not self.has_next():
                    return None
                addr = self.dataset[self.batch_idx][self.table_idx][self.vec_idx]
                self._advance()
                return addr
            
            def _advance(self):
                self.vec_idx += 1
                if self.vec_idx >= len(self.dataset[self.batch_idx][self.table_idx]):
                    self.vec_idx = 0
                    self.table_idx += 1
                    if self.table_idx >= self.table_end:
                        self.table_idx = self.table_start
                        self.batch_idx += 1
        
        iterators = [
            SimpleIterator(emb_dataset, table_start, table_end)
            for table_start, table_end in table_ranges
        ]
        
        # Round-robin collection
        flat_list = []
        tick = 0
        active = [True] * self.num_cores
        
        while any(active):
            core_id = tick % self.num_cores
            if active[core_id] and iterators[core_id].has_next():
                addr = iterators[core_id].get_next()
                if addr is not None:
                    flat_list.append(addr)
                if not iterators[core_id].has_next():
                    active[core_id] = False
            tick += 1
        
        return np.array(flat_list, dtype=np.int64)

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
    def __init__(self, mem_size, mem_gran, emb_dim, n_format_byte, emb_dataset, vectors_per_table, prof_multiplier, spad_policy, num_cores=1, debug=False):
        self.mem_size = mem_size
        self.mem_gran = mem_gran
        self.emb_dim = emb_dim
        self.n_format_byte = n_format_byte
        self.emb_dataset = emb_dataset
        self.vectors_per_table = vectors_per_table
        self.prof_multiplier = prof_multiplier
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
                    for v_i in range(self.vectors_per_table):
                        for d_i in range(self.access_per_vector):
                            bytes_per_vec = (self.emb_dim * self.n_format_byte - 1).bit_length()
                            tbl_bits = t_i << int(np.log2(self.vectors_per_table-1)+1 + bytes_per_vec)
                            vec_idx = v_i << bytes_per_vec
                            dim_bits = self.mem_gran * d_i
                            this_addr = tbl_bits + vec_idx + dim_bits
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
