import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box
from policies import LRUPolicy, SRRIPPolicy, OptPolicy, ProfilePolicy, SpadPolicy

class CoreAccessIterator:
    """Iterator for a core's memory accesses in table-partitioned multicore simulation"""
    def __init__(self, emb_dataset, table_start, table_end, offmem_trace):
        self.emb_dataset = emb_dataset
        self.table_start = table_start
        self.table_end = table_end
        self.offmem_trace = offmem_trace
        
        self.batch_idx = 0
        self.table_idx = table_start
        self.vec_idx = 0
        
        self.total_accesses = self._count_total_accesses()
        self.current_access = 0
    
    def _count_total_accesses(self):
        """Count total number of accesses this core will make"""
        total = 0
        for batch in self.emb_dataset:
            for table_id in range(self.table_start, self.table_end):
                total += len(batch[table_id])
        return total
    
    def has_next(self):
        """Check if iterator has more accesses"""
        return self.current_access < self.total_accesses
    
    def get_next(self):
        """Get next memory address and its location"""
        if not self.has_next():
            return None
        
        addr = self.emb_dataset[self.batch_idx][self.table_idx][self.vec_idx]
        result = (self.batch_idx, self.table_idx, self.vec_idx, addr)
        
        self._advance()
        return result
    
    def _advance(self):
        """Move to next vector/table/batch"""
        self.vec_idx += 1
        self.current_access += 1
        
        if self.vec_idx >= len(self.emb_dataset[self.batch_idx][self.table_idx]):
            self.vec_idx = 0
            self.table_idx += 1
            
            if self.table_idx >= self.table_end:
                self.table_idx = self.table_start
                self.batch_idx += 1
                
                if self.batch_idx >= len(self.emb_dataset):
                    self.batch_idx = len(self.emb_dataset) - 1


class CoreOnmem:
    def __init__(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table=0, mem_gran=0, prof_multiplier=1, mem_latency=1, num_cores=1):
        self.mem_size = 0
        self.mem_type = "init"
        self.mem_policy = "init"
        self.mem_latency = 1
        self.on_mem = np.ones(1)
        self.batch_counter = 0
        self.profile_filter = np.ones(1)
        
        self.emb_dim = 0
        self.emb_dataset = np.ones(1)
        
        self.cache_way = 0
        self.cache_line_size = 0
        self.cache_set = 0
        self.cache_tag_bits = 0
        self.n_format_byte = 0        
        self.rrpv_bits = 0
        self.rrpv_insert = 0
        
        self.mem_gran = mem_gran
        self.prof_multiplier = prof_multiplier
        self.vectors_per_table = 0

        self.access_results = []
        self.spad_load_results = []
        
        # Multicore configuration
        self.num_cores = num_cores
        self.core_access_results = [[] for _ in range(num_cores)]
        
        self.set_params(mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table, mem_gran, prof_multiplier, mem_latency)
        
    def set_params(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table=0, mem_gran=0, prof_multiplier=1, mem_latency=1):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        self.mem_latency = mem_latency  # Access latency in cycles
        
        # below configs are related to the dataset
        self.emb_dim = emb_dim
        self.emb_dataset = emb_dataset
        
        self.n_format_byte = n_format_byte
        
        if self.mem_type == "cache":
            # below configs are only for cache configurations
            self.cache_way = cache_config[0] # cache_config = [way, line size]
            self.cache_line_size = cache_config[1]
            self.cache_set = int(self.mem_size / self.cache_line_size / self.cache_way)
            # Fix: Use ceiling of log2 to handle non-power-of-2 cache sets
            self.cache_index_bits = int(np.ceil(np.log2(self.cache_set))) if self.cache_set > 1 else 0
            self.cache_offset_bits = int(np.log2(self.cache_line_size-1)+1) # byte offset
            self.cache_tag_bits = 48 - self.cache_index_bits - self.cache_offset_bits # 48 bits - index bits - byte offset
            self.rrpv_bits = cache_config[2]
            self.rrpv_insert = cache_config[3]
        elif self.mem_type == "spad":
            self.mem_gran = mem_gran
            self.prof_multiplier = prof_multiplier
            self.vectors_per_table = vectors_per_table

        self.offmem_trace = [[np.full_like(self.emb_dataset[nb][nt], -1) for nt in range(len(self.emb_dataset[nb]))] for nb in range(len(self.emb_dataset))]
        print("[DEBUG] self.offmem_trace shape: ({}, {}, {})".format(len(self.offmem_trace), len(self.offmem_trace[0]), len(self.offmem_trace[0][0])))
    
    def _partition_tables_across_cores(self):
        """Partition embedding tables across multiple cores"""
        total_tables = len(self.emb_dataset[0])
        tables_per_core = total_tables // self.num_cores
        remainder = total_tables % self.num_cores
        
        table_ranges = []
        start_idx = 0
        
        for core_id in range(self.num_cores):
            extra = 1 if core_id < remainder else 0
            end_idx = start_idx + tables_per_core + extra
            table_ranges.append((start_idx, end_idx))
            start_idx = end_idx
        
        return table_ranges

    def set_policy(self, policy):
        self.mem_policy = policy
        if self.mem_type == "cache":
            if not policy.startswith("cache_"):
                assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
            cache_config = [self.cache_way, self.cache_line_size, self.rrpv_bits, self.rrpv_insert, self.cache_set]
            if self.mem_policy == "cache_LRU":
                self.policy = LRUPolicy(cache_config)
            elif self.mem_policy == "cache_SRRIP":
                self.policy = SRRIPPolicy(cache_config)
            elif self.mem_policy == "cache_OPT":
                self.policy = OptPolicy(cache_config, self.emb_dataset, self.num_cores)
            elif self.mem_policy == "cache_profile":
                self.policy = ProfilePolicy(cache_config, self.emb_dataset, self.batch_counter)
            else:
                raise NotImplementedError(f"Policy {self.mem_policy} not implemented")
        elif self.mem_type == "spad":
            if not policy.startswith("spad_"):
                assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
            self.policy = SpadPolicy(self.mem_size, self.mem_gran, self.emb_dim, self.n_format_byte, self.emb_dataset, self.vectors_per_table, self.prof_multiplier, self.mem_policy, self.num_cores)

        self.policy.initialize()
        self.on_mem = self.policy.on_mem

    def print_config(self):
        content = [
            f"Number of cores: {self.num_cores}",
            f"Memory size: {self.mem_size} B ({int(self.mem_size/1024/1024)} MB)",
            f"Memory type: {self.mem_type}",
            f"Memory policy: {self.mem_policy}",
            f"Memory access latency: {self.mem_latency} cycles"
        ]
        if self.mem_type == "cache":
            content.extend([
                f"Cache way: {self.cache_way}-way",
                f"Cache line size: {self.cache_line_size} B",
                f"Cache set: {self.cache_set} sets",
                f"Cache tag bits: {self.cache_tag_bits} bits"
            ])
        print_styled_box("On-Chip Memory Configuration", content)

    def print_sim(self):
        print_styled_header("Simulation Start")

    def get_tag_bits(self, addr):
        # make bits lower than tag bits to zero
        tag_mask_bits = (1 << (self.cache_index_bits + self.cache_offset_bits)) - 1
        return addr & ~tag_mask_bits

    def get_index_bits(self, addr):
        if self.cache_index_bits == 0:
            return 0
        
        index_msb = self.cache_index_bits + self.cache_offset_bits - 1
        index_lsb = self.cache_offset_bits
        mask = ((1 << (index_msb - index_lsb + 1)) - 1) << index_lsb
        index_bits = (addr & mask) >> index_lsb    # extract only index bits
        
        # Ensure index is within bounds for non-power-of-2 cache sets, this is just for 384 MB cache simulation.
        return index_bits % self.cache_set

    def do_simulation(self):
        self.print_sim()
        print(f"Running with {self.num_cores} core(s)")
        
        # Partition tables across cores
        table_ranges = self._partition_tables_across_cores()
        for core_id in range(self.num_cores):
            table_start, table_end = table_ranges[core_id]
            print(f"[Core {core_id}] Tables {table_start}-{table_end-1}")
        
        # Create access iterator for each core
        core_iterators = []
        for core_id in range(self.num_cores):
            table_start, table_end = table_ranges[core_id]
            iterator = CoreAccessIterator(
                self.emb_dataset,
                table_start,
                table_end,
                self.offmem_trace
            )
            core_iterators.append(iterator)
        
        # Initialize per-core statistics
        core_batch_stats = [[0, 0] for _ in range(self.num_cores)]
        total_accesses = sum(it.total_accesses for it in core_iterators)
        
        if self.mem_type == 'spad':
            # SPad: static allocation, no inter-core contention
            # Process each core independently with vectorized operations
            if isinstance(self.on_mem, set):
                on_mem_array = np.array(list(self.on_mem), dtype=np.int64)
            else:
                on_mem_array = self.on_mem
            
            with tqdm(total=len(self.emb_dataset) * self.num_cores, desc="Simulation") as pbar:
                for core_id in range(self.num_cores):
                    table_start, table_end = table_ranges[core_id]
                    
                    for nb in range(len(self.emb_dataset)):
                        for nt in range(table_start, table_end):
                            hit_mask = np.isin(self.emb_dataset[nb][nt], on_mem_array)
                            
                            num_hit = np.sum(hit_mask)
                            num_miss = np.sum(~hit_mask)
                            
                            core_batch_stats[core_id][0] += num_hit
                            core_batch_stats[core_id][1] += num_miss
                            
                            miss_mask = ~hit_mask
                            self.offmem_trace[nb][nt][miss_mask] = self.emb_dataset[nb][nt][miss_mask]
                        
                        pbar.update(1)
        
        elif self.mem_type == 'cache':
            # Cache: shared state, round-robin needed for inter-core contention
            tick = 0
            active_cores = [True] * self.num_cores
            
            with tqdm(total=total_accesses, desc="Simulation") as pbar:
                while any(active_cores):
                    core_id = tick % self.num_cores
                    
                    if active_cores[core_id] and core_iterators[core_id].has_next():
                        batch_id, table_id, vec_id, addr = core_iterators[core_id].get_next()
                        
                        tag = self.get_tag_bits(addr)
                        index = self.get_index_bits(addr)
                        
                        hit, victim = self.policy.handle_access(tag, index)
                        
                        if hit:
                            core_batch_stats[core_id][0] += 1
                        else:
                            core_batch_stats[core_id][1] += 1
                            self.offmem_trace[batch_id][table_id][vec_id] = addr
                        
                        self.policy.post_access_processing(hit, tag, index, vec_id)
                        pbar.update(1)
                        
                        if not core_iterators[core_id].has_next():
                            active_cores[core_id] = False
                    
                    tick += 1
        
        # Aggregate results
        for core_id in range(self.num_cores):
            self.core_access_results[core_id].append(core_batch_stats[core_id])
        
        total_hits = sum(stats[0] for stats in core_batch_stats)
        total_miss = sum(stats[1] for stats in core_batch_stats)
        self.access_results.append([total_hits, total_miss])
        
        # Verify offmem_trace integrity
        offmem_count = 0
        for nb in range(len(self.offmem_trace)):
            for nt in range(len(self.offmem_trace[nb])):
                offmem_count += np.sum(self.offmem_trace[nb][nt] != -1)
        
        print(f"[DEBUG] Total misses: {total_miss}, Offmem trace entries: {offmem_count}")
        if offmem_count != total_miss:
            print(f"[WARNING] Mismatch between miss count and offmem_trace entries!")
        
        print("Simulation Done")
        self.print_stats()
        
    def print_stats(self):
        total_hits = 0
        total_miss = 0
        for i in range(len(self.access_results)):
            total_hits = total_hits + self.access_results[i][0]
            total_miss = total_miss + self.access_results[i][1]
        total_hit_ratio = total_hits / (total_hits + total_miss)
        
        content = [
            f"Total hit ratio: {total_hit_ratio:.4f}",
            f"Total accesses: {total_hits+total_miss}",
            f"Total hits: {total_hits}",
            f"Total misses: {total_miss}",
            "----------------------------------------",
            "Per-Core Statistics:"
        ]
        
        # Per-core statistics
        for core_id in range(self.num_cores):
            core_hits = sum(batch[0] for batch in self.core_access_results[core_id])
            core_miss = sum(batch[1] for batch in self.core_access_results[core_id])
            core_total = core_hits + core_miss
            core_hit_ratio = core_hits / core_total if core_total > 0 else 0
            
            content.append(
                f"[Core {core_id}] Hit ratio: {core_hit_ratio:.4f}   "
                f"Accesses: {core_total}   Hits: {core_hits}   Misses: {core_miss}"
            )
        
        print_styled_box("Simulation Results", content)