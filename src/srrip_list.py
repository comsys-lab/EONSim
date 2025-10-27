import numpy as np

class SRRIPList:
    """Hash map + list based SRRIP cache for O(1) tag lookup"""
    
    def __init__(self, cache_way, rrpv_bits, rrpv_insert, num_sets=1):
        """
        Initialize SRRIP cache
        
        Args:
            cache_way: Number of ways (entries per set)
            rrpv_bits: Number of bits for RRPV counter
            rrpv_insert: Initial RRPV value for new entries
            num_sets: Number of cache sets (default 1 for fully associative)
        """
        self.cache_way = cache_way
        self.rrpv_bits = rrpv_bits
        self.rrpv_insert = rrpv_insert
        self.max_rrpv = (1 << rrpv_bits) - 1
        self.num_sets = num_sets
        
        # Each set: list of [tag, rrpv] pairs and dict for O(1) lookup
        self.cache_sets = [[] for _ in range(num_sets)]
        self.tag_to_idx = [{}for _ in range(num_sets)]
    
    def get_set_index(self, addr):
        """Get set index from address (fully associative for now)"""
        return 0
    
    def access(self, tag):
        """
        Access cache with given tag
        
        Args:
            tag: Memory address tag
            
        Returns:
            True if cache hit, False if miss
        """
        set_idx = self.get_set_index(tag)
        cache_set = self.cache_sets[set_idx]
        tag_map = self.tag_to_idx[set_idx]
        
        # Check for cache hit using hash map
        if tag in tag_map:
            idx = tag_map[tag]
            cache_set[idx][1] = 0  # Reset RRPV to 0 on hit
            return True
        
        # Cache miss - insert new entry
        if len(cache_set) < self.cache_way:
            # Still have space, just add
            new_idx = len(cache_set)
            cache_set.append([tag, self.rrpv_insert])
            tag_map[tag] = new_idx
        else:
            # Need to evict - find victim with max RRPV
            replaced = False
            
            while not replaced:
                # Find first entry with max RRPV
                victim_idx = None
                for i, entry in enumerate(cache_set):
                    if entry[1] >= self.max_rrpv:
                        victim_idx = i
                        break
                
                if victim_idx is not None:
                    # Evict victim and insert new entry
                    old_tag = cache_set[victim_idx][0]
                    del tag_map[old_tag]
                    cache_set[victim_idx] = [tag, self.rrpv_insert]
                    tag_map[tag] = victim_idx
                    replaced = True
                else:
                    # No victim found, age all entries
                    for entry in cache_set:
                        if entry[1] < self.max_rrpv:
                            entry[1] += 1
        
        return False
    
    def get_entries(self):
        """Return all entries from set 0 as numpy array"""
        cache_set = self.cache_sets[0]
        result = np.zeros(self.cache_way, dtype=np.int64)
        
        for i, entry in enumerate(cache_set):
            if i < self.cache_way:
                result[i] = entry[0]
        
        return result
    
    def is_empty(self):
        """Check if cache is empty"""
        return len(self.cache_sets[0]) == 0
    
    def get_set_entries(self, set_idx):
        """
        Get entries for a specific set with their RRPV values
        
        Args:
            set_idx: Index of the cache set
            
        Returns:
            numpy array of shape (cache_way, 2) with [tag, rrpv] pairs
        """
        cache_set = self.cache_sets[set_idx]
        result = np.zeros((self.cache_way, 2), dtype=np.int64)
        
        for i, entry in enumerate(cache_set):
            if i < self.cache_way:
                result[i, 0] = entry[0]  # tag
                result[i, 1] = entry[1]  # RRPV
        
        return result
    
    def get_num_entries(self, set_idx):
        """Get number of valid entries in a set"""
        return len(self.cache_sets[set_idx])
    
    def get_num_sets(self):
        """Get total number of sets"""
        return self.num_sets
