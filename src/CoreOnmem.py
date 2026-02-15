import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box
from policies import LRUPolicy, SRRIPPolicy, OptPolicy, ProfilePolicy, SpadPolicy

class CoreOnmem:
    def __init__(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte, vectors_per_table=0, mem_gran=0, prof_multiplier=1, mem_latency=1):
        self.mem_size = 0 # KB
        self.mem_type = "init"
        self.mem_policy = "init"
        self.mem_latency = 1  # Access latency in cycles (default: 1)
        self.on_mem = np.ones(1)
        self.batch_counter = 0 # this is only for cache_profile
        self.profile_filter = np.ones(1) # this is only for cache_profile
        
        # below configs are related to the dataset
        self.emb_dim = 0 # this is for spad
        self.emb_dataset = np.ones(1)
        
        # below configs are only for cache configurations
        self.cache_way = 0
        self.cache_line_size = 0
        self.cache_set = 0
        self.cache_tag_bits = 0
        self.n_format_byte = 0        
        self.rrpv_bits = 0
        self.rrpv_insert = 0
        
        # SPM specific
        self.mem_gran = mem_gran
        self.prof_multiplier = prof_multiplier
        self.vectors_per_table = 0 # This will be set in set_params

        self.access_results = []
        self.spad_load_results = []
        
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

        # Initialize offmem_trace with same structure as emb_dataset (storing off-chip memory access trace with -1 init)
        self.offmem_trace = [[np.full_like(self.emb_dataset[nb][nt], -1) for nt in range(len(self.emb_dataset[nb]))] for nb in range(len(self.emb_dataset))]
        print("[DEBUG] self.offmem_trace shape: ({}, {}, {})".format(len(self.offmem_trace), len(self.offmem_trace[0]), len(self.offmem_trace[0][0])))

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
                self.policy = OptPolicy(cache_config, self.emb_dataset)
            elif self.mem_policy == "cache_profile":
                self.policy = ProfilePolicy(cache_config, self.emb_dataset, self.batch_counter)
            else:
                raise NotImplementedError(f"Policy {self.mem_policy} not implemented")
        elif self.mem_type == "spad":
            if not policy.startswith("spad_"):
                assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
            self.policy = SpadPolicy(self.mem_size, self.mem_gran, self.emb_dim, self.n_format_byte, self.emb_dataset, self.vectors_per_table, self.prof_multiplier, self.mem_policy)

        self.policy.initialize()
        self.on_mem = self.policy.on_mem

    def print_config(self):
        content = [
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
        for nb in range(len(self.emb_dataset)):
            num_hit = 0
            num_miss = 0
            num_spad_load = 0

            print(f"Processing batch {nb}...")
            
            if self.mem_type == 'spad':
                with tqdm(total=len(self.emb_dataset[nb]), desc="Simulation") as pbar:
                    for nt in range(len(self.emb_dataset[nb])):
                        # Convert on_mem set to numpy array for vectorized operations
                        if isinstance(self.on_mem, set):
                            on_mem_array = np.array(list(self.on_mem), dtype=np.int64)
                        else:
                            on_mem_array = self.on_mem

                        # Use vectorized isin operation to find hits
                        hit_mask = np.isin(self.emb_dataset[nb][nt], on_mem_array)
                        
                        # Count hits and misses
                        num_hit += np.sum(hit_mask)
                        num_miss += np.sum(~hit_mask)
                        
                        # Update offmem_trace for misses - vectorized operation
                        miss_mask = ~hit_mask
                        self.offmem_trace[nb][nt][miss_mask] = self.emb_dataset[nb][nt][miss_mask]
                        
                        pbar.update(1)
                
                if self.mem_policy == "spad_oracle":
                    self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset)-1)
                    self.policy.batch_counter = self.batch_counter
                    if self.batch_counter % self.prof_multiplier == 0:
                        self.on_mem = self.policy.set_spad()
                        num_spad_load += self.policy.spad_size

            elif self.mem_type == 'cache':
                with tqdm(total=sum(len(table) for table in self.emb_dataset[nb]), desc="Processing") as pbar:
                    for nt in range(len(self.emb_dataset[nb])):
                        hit_mask = np.zeros(len(self.emb_dataset[nb][nt]), dtype=bool)

                        for vec in range(len(self.emb_dataset[nb][nt])):
                            this_addr = self.emb_dataset[nb][nt][vec]
                            this_tag = self.get_tag_bits(this_addr)
                            this_index = self.get_index_bits(this_addr)

                            hit, _ = self.policy.handle_access(this_tag, this_index)

                            if hit:
                                num_hit += 1
                                hit_mask[vec] = True
                            else:
                                num_miss += 1

                            self.policy.post_access_processing(hit, this_tag, this_index, vec)
                            pbar.update(1)

                        miss_mask = ~hit_mask
                        self.offmem_trace[nb][nt][miss_mask] = self.emb_dataset[nb][nt][miss_mask]

            self.access_results.append([num_hit, num_miss])
            if self.mem_type == 'spad':
                self.spad_load_results.append(num_spad_load)

            if self.mem_policy == "cache_profile":
                self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset) - 1)
                self.policy.batch_counter = self.batch_counter
                self.policy.profile_filter = self.policy.create_profile_filter()

        print("Simulation Done")
        self.print_stats()
        
    def print_stats(self):
        # calculate total results
        total_hits = 0
        total_miss = 0
        for i in range(len(self.access_results)):
            total_hits = total_hits + self.access_results[i][0]
            total_miss = total_miss + self.access_results[i][1]
        total_hit_ratio = total_hits / (total_hits + total_miss)
        
        # prepare content for styled box
        content = [
            f"Total hit ratio: {total_hit_ratio:.4f}",
            f"Total accesses: {total_hits+total_miss}",
            f"Total hits: {total_hits}",
            f"Total misses: {total_miss}",
            "----------------------------------------",
            "Per batch results"
        ]
        
        # add per batch results
        for i in range(len(self.access_results)):
            batch_hit_ratio = self.access_results[i][0] / (self.access_results[i][0] + self.access_results[i][1])
            content.append(
                f"[Batch {i}] hit ratio: {batch_hit_ratio:.4f}   accesses: {self.access_results[i][0]+self.access_results[i][1]}   hits: {self.access_results[i][0]}   misses: {self.access_results[i][1]}"
            )
            
            if self.mem_type == 'spad' and i < len(self.spad_load_results):
                content.append(
                    f"[Batch {i}] spad load: {self.spad_load_results[i]}"
                )
        
        
        print_styled_box("Simulation Results", content)