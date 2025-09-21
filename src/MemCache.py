import numpy as np
import time
import torch
import itertools
import random
from collections import OrderedDict, Counter
from LRUlist import LRUlist
from tqdm import tqdm
from Helper import print_styled_header, print_styled_box

class MemCache:
    def __init__(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte):
        self.mem_size = 0 # KB
        self.mem_type = "init"
        self.mem_policy = "init"
        self.on_mem = np.ones(1)
        self.batch_counter = 0 # this is only for cache_profile
        self.profile_filter = np.ones(1) # this is only for cache_profile
        
        # below configs are related to the dataset
        self.emb_dim = 0 # this is for spad
        self.emb_dataset = np.ones(1)
        self.flat_dataset = np.ones(1)
        
        # below configs are only for cache configurations
        self.cache_way = 0
        self.cache_line_size = 0
        self.cache_set = 0
        self.cache_tag_bits = 0
        self.n_format_byte = 0        
        self.rrpv_bits = 0
        self.rrpv_insert = 0
        
        self.access_results = []
               
        self.set_params(mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte)
        
    def set_params(self, mem_size, mem_type, cache_config, emb_dim, emb_dataset, n_format_byte):
        self.mem_size = mem_size * 1024 # KB -> Byte
        self.mem_type = mem_type # spad or cache
        
        # below configs are related to the dataset
        self.emb_dim = emb_dim # this is for spad
        self.emb_dataset = emb_dataset # emb_dataset[numbatch][table][batchsz*embdim]
        
        # below configs are only for cache configurations
        self.cache_way = cache_config[0] # cache_config = [way, line size]
        self.cache_line_size = cache_config[1]
        self.cache_set = int(self.mem_size / self.cache_line_size / self.cache_way)
        self.cache_index_bits = int(np.log2(self.cache_set-1)+1)
        self.cache_offset_bits = int(np.log2(self.cache_line_size-1)+1) # byte offset
        self.cache_tag_bits = 48 - self.cache_index_bits - self.cache_offset_bits # 48 bits - index bits - byte offset
        self.rrpv_bits = cache_config[2]
        self.rrpv_insert = cache_config[3]
        
        self.n_format_byte = n_format_byte
        
    def set_policy(self, policy):
        if (self.mem_type == "cache" and not policy.startswith("cache_")):
            assert False, f"Invalid policy: '{policy}' for mem_type: '{self.mem_type}'"
        self.mem_policy = policy        
        
    def print_config(self):
        content = [
            f"Memory size: {self.mem_size} B ({int(self.mem_size/1024/1024)} MB)",
            f"Memory type: {self.mem_type}",
            f"Memory policy: {self.mem_policy}",
            f"Cache way: {self.cache_way}-way",
            f"Cache line size: {self.cache_line_size} B",
            f"Cache set: {self.cache_set} sets",
            f"Cache tag bits: {self.cache_tag_bits} bits"
        ]
        print_styled_box("On-Chip Memory Configuration", content)
        
    def print_sim(self):
        print_styled_header("Simulation Start")
        
    def get_tag_bits(self, addr):
        # make bits lower than tag bits to zero
        tag_mask_bits = (1 << (self.cache_index_bits + self.cache_offset_bits)) - 1
        return addr & ~tag_mask_bits
    
    def get_index_bits(self, addr):
        index_msb = self.cache_index_bits + self.cache_offset_bits - 1
        index_lsb = self.cache_offset_bits
        mask = ((1 << (index_msb - index_lsb + 1)) - 1) << index_lsb
        index_bits = (addr & mask) >> index_lsb    # extract only index bits
        return index_bits
    
    def OPT_replacement(self, curr_cycle, this_index):
        indices_t = np.array([], dtype=np.int64)
        # print(self.on_mem[this_index].shape)
        for tag in self.on_mem[this_index]:
            ind = np.where(self.flat_dataset[curr_cycle:] == tag)[0]
            if len(ind) == 0:
                indices_t = np.append(indices_t, -1)
            else:
                indices_t = np.append(indices_t, ind[0])
        
        if -1 in indices_t:
            evict_index = np.argmax(indices_t == -1)
        else:
            evict_index = np.argmax(indices_t)
        return evict_index
    
    def create_on_mem(self):
        # create on-chip memory data structure
        if self.mem_policy == "cache_LRU" or self.mem_policy == "cache_profile":
            self.on_mem = [LRUlist(self.cache_way) for i in range(self.cache_set)]
        elif self.mem_policy == "cache_OPT":
            self.on_mem = [np.array([], dtype=np.int64) for i in range(self.cache_set)]
        elif self.mem_policy == "cache_SRRIP":
            # Initialize as 2D arrays with [tag, RRPV] pairs
            self.on_mem = [np.zeros((0, 2), dtype=np.int64) for i in range(self.cache_set)]
            
        if self.mem_policy == "cache_profile":
            # flatten the dataset -> count and sort the access frequency of each memory address
            self.flat_dataset = itertools.chain.from_iterable(self.emb_dataset[self.batch_counter])
            access_freq = Counter(self.flat_dataset)      
            access_freq = access_freq.most_common()
            new_access_freq = [(key, value) for key, value in access_freq if value >=3]
            self.profile_filter = np.array([x[0] for x in new_access_freq], dtype = np.int64)
            # print(len(access_freq))
            # print(len(new_access_freq))
            # print(access_freq[int(len(access_freq)*0.2)])
            # exit()
        if self.mem_policy == "cache_OPT":
            self.flat_dataset = itertools.chain.from_iterable(self.emb_dataset)
            self.flat_dataset = np.array(list(self.flat_dataset), dtype = np.int64).flatten()
            self.flat_dataset = self.get_tag_bits(self.flat_dataset)
            
    def do_simulation(self):
        # Simulation
        self.print_sim()
        if self.mem_policy == "cache_profile":
            self.do_simulation_profile()
        elif self.mem_policy == "cache_OPT":
            self.do_simulation_OPT()
        elif self.mem_policy == "cache_SRRIP":
            self.do_simulation_SRRIP()
        else:
            for nb in range(len(self.emb_dataset)): # recall that self.emb_dataset[numbatch][table][batchsz*lookuppersample]
                num_hit = 0
                num_miss = 0
                
                print("Processing batch {}...".format(nb))
                with tqdm(total=len(self.emb_dataset[nb])*len(self.emb_dataset[nb][0]), desc="Processing") as pbar:
                    for nt in range(len(self.emb_dataset[nb])):
                        for vec in range(len(self.emb_dataset[nb][nt])):
                            this_tag = self.get_tag_bits(self.emb_dataset[nb][nt][vec])
                            this_index = self.get_index_bits(self.emb_dataset[nb][nt][vec])
                            # print("[DEBUG] this_addr:{}   this_index:{}   this_tag: {}".format(self.emb_dataset[nb][nt][vec], this_index, this_tag))
                            if self.on_mem[this_index].search_and_access(this_tag): # tag matching
                                num_hit = num_hit + 1
                            else:
                                self.on_mem[this_index].insert_node(this_tag)
                                num_miss = num_miss + 1
                            # self.on_mem[this_index].print_list()
                            pbar.update(1)
                        
                
                self.access_results.append([num_hit, num_miss]) # add the results for each batch
            
        print("Simulation Done")
        self.print_stats()
    
    def do_simulation_SRRIP(self):
        for nb in range(len(self.emb_dataset)):
            num_hit = 0
            num_miss = 0
            
            print("Processing batch {}...".format(nb))
            with tqdm(total=len(self.emb_dataset[nb])*len(self.emb_dataset[nb][0]), desc="Processing") as pbar:
                for nt in range(len(self.emb_dataset[nb])):
                    for vec in range(len(self.emb_dataset[nb][nt])):
                        this_tag = self.get_tag_bits(self.emb_dataset[nb][nt][vec])
                        this_index = self.get_index_bits(self.emb_dataset[nb][nt][vec])
                        
                        # Check if tag exists in cache
                        tag_match = np.where(self.on_mem[this_index][:,0] == this_tag)[0]
                        
                        if len(tag_match) > 0: # Cache hit
                            num_hit += 1
                            # Update RRPV to 0 on hit
                            self.on_mem[this_index][tag_match[0], 1] = 0
                        else: # Cache miss
                            num_miss += 1
                            if len(self.on_mem[this_index]) < self.cache_way:
                                # Add new entry with RRPV_insert
                                new_entry = np.array([[this_tag, self.rrpv_insert]])
                                self.on_mem[this_index] = np.vstack([self.on_mem[this_index], new_entry])
                            else:
                                max_rrpv = 2**self.rrpv_bits - 1
                                replaced = False
                                
                                while not replaced:
                                    # Find entries with max RRPV
                                    victim_candidates = np.where(self.on_mem[this_index][:,1] == max_rrpv)[0]
                                    
                                    if len(victim_candidates) > 0:
                                        # Replace first victim found
                                        self.on_mem[this_index][victim_candidates[0]] = [this_tag, self.rrpv_insert]
                                        replaced = True
                                    else:
                                        # Increment all RRPV values
                                        self.on_mem[this_index][:,1] = np.minimum(
                                            self.on_mem[this_index][:,1] + 1, 
                                            max_rrpv
                                        )
                        
                        pbar.update(1)
            
            self.access_results.append([num_hit, num_miss])
    
    def do_simulation_OPT(self):
        for nb in range(len(self.emb_dataset)): # recall that self.emb_dataset[numbatch][table][batchsz*lookuppersample]
            num_hit = 0
            num_miss = 0
            curr_cycle = 0
            
            print("Processing batch {}...".format(nb))
            with tqdm(total=len(self.emb_dataset[nb])*len(self.emb_dataset[nb][0]), desc="Processing") as pbar:
                for nt in range(len(self.emb_dataset[nb])):
                    for vec in range(len(self.emb_dataset[nb][nt])):
                        this_tag = self.get_tag_bits(self.emb_dataset[nb][nt][vec])
                        this_index = self.get_index_bits(self.emb_dataset[nb][nt][vec])
                        
                        if this_tag in self.on_mem[this_index]: # cache hit
                            num_hit = num_hit + 1
                        else: # cache miss
                            num_miss = num_miss + 1
                            if len(self.on_mem[this_index]) < (self.cache_way): # there is an empty way
                                self.on_mem[this_index] = np.append(self.on_mem[this_index], this_tag)
                            else: # there is an empty way, replacement is required
                                evict_index = self.OPT_replacement(curr_cycle, this_index)
                                self.on_mem[this_index][evict_index] = this_tag # do replacement
                                # self.on_mem[this_index] = np.append(self.on_mem[this_index], this_tag)
                        
                        curr_cycle = curr_cycle + 1
                        pbar.update(1)
                    # print("DEBUG"+str(len(self.on_mem)))
                    # print("DEBUG"+str(self.on_mem[0].shape))
            
            self.access_results.append([num_hit, num_miss]) # add the results for each batch
    
    def do_simulation_profile(self):
        # Simulation
        for nb in range(len(self.emb_dataset)): # recall that self.emb_dataset[numbatch][table][batchsz*lookuppersample]
            num_hit = 0
            num_miss = 0
            
            print("Processing batch {}...".format(nb))
            with tqdm(total=len(self.emb_dataset[nb])*len(self.emb_dataset[nb][0]), desc="Processing") as pbar:
                for nt in range(len(self.emb_dataset[nb])):
                    for vec in range(len(self.emb_dataset[nb][nt])):
                        # if not np.isin(self.emb_dataset[nb][nt][vec], self.profile_filter):
                        if not self.emb_dataset[nb][nt][vec] in self.profile_filter:
                            num_miss = num_miss + 1
                            pbar.update(1)
                            continue
                        else:
                            this_tag = self.get_tag_bits(self.emb_dataset[nb][nt][vec])
                            this_index = self.get_index_bits(self.emb_dataset[nb][nt][vec])
                            if self.on_mem[this_index].search_and_access(this_tag): # tag matching
                                num_hit = num_hit + 1
                            else:
                                self.on_mem[this_index].insert_node(this_tag)
                                num_miss = num_miss + 1
                            pbar.update(1)
            self.batch_counter = min(self.batch_counter + 1, len(self.emb_dataset)-1)
            self.create_on_mem()
            
            self.access_results.append([num_hit, num_miss]) # add the results for each batch
        
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
                f"[Batch {i}] hit ratio: {batch_hit_ratio:.4f}   accesses: {self.access_results[i][0]+self.access_results[i][1]}   "
                f"hits: {self.access_results[i][0]}   misses: {self.access_results[i][1]}"
            )
        
        print_styled_box("Simulation Results", content)