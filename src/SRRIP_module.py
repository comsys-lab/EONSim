import numpy as np

class SRRIPList:
    """Hash map + list based SRRIP cache for O(1) tag lookup"""
    
    def __init__(self, cache_way, rrpv_bits, rrpv_insert):
        """
        Initialize SRRIP cache
        
        Args:
            cache_way: Number of ways (entries per set)
            rrpv_bits: Number of bits for RRPV counter
            rrpv_insert: Initial RRPV value for new entries
        """
        self.cache_way = cache_way
        self.rrpv_bits = rrpv_bits
        self.rrpv_insert = rrpv_insert
        self.max_rrpv = (1 << rrpv_bits) - 1
        
        # List of [tag, rrpv] pairs and dict for O(1) lookup
        self.cache_entries = []
        self.tag_to_idx = {}
    
    def access(self, tag):
        """
        Access cache with given tag
        
        Args:
            tag: Memory address tag
            
        Returns:
            True if cache hit, False if miss
        """
        # Check for cache hit using hash map
        if tag in self.tag_to_idx:
            idx = self.tag_to_idx[tag]
            self.cache_entries[idx][1] = 0  # Reset RRPV to 0 on hit
            return True
        
        # Cache miss - insert new entry
        if len(self.cache_entries) < self.cache_way:
            # Still have space, just add
            new_idx = len(self.cache_entries)
            self.cache_entries.append([tag, self.rrpv_insert])
            self.tag_to_idx[tag] = new_idx
        else:
            # Need to evict - find victim with max RRPV
            replaced = False
            
            while not replaced:
                # Find first entry with max RRPV
                victim_idx = None
                for i, entry in enumerate(self.cache_entries):
                    if entry[1] >= self.max_rrpv:
                        victim_idx = i
                        break
                
                if victim_idx is not None:
                    # Evict victim and insert new entry
                    old_tag = self.cache_entries[victim_idx][0]
                    del self.tag_to_idx[old_tag]
                    self.cache_entries[victim_idx] = [tag, self.rrpv_insert]
                    self.tag_to_idx[tag] = victim_idx
                    replaced = True
                else:
                    # No victim found, age all entries
                    for entry in self.cache_entries:
                        if entry[1] < self.max_rrpv:
                            entry[1] += 1
        
        return False
    
    def get_entries(self):
        """Return all entries as numpy array of shape (cache_way, 2) with [tag, rrpv]"""
        result = np.zeros((self.cache_way, 2), dtype=np.int64)
        
        for i, entry in enumerate(self.cache_entries):
            if i < self.cache_way:
                result[i, 0] = entry[0]  # tag
                result[i, 1] = entry[1]  # RRPV
        
        return result
    
    def is_empty(self):
        """Check if cache is empty"""
        return len(self.cache_entries) == 0
    
    def get_num_entries(self):
        """Get number of valid entries"""
        return len(self.cache_entries)
    
    def print_list(self):
        """Print cache contents for debugging"""
        for i, entry in enumerate(self.cache_entries):
            print(f"[{i}] tag: {entry[0]}, RRPV: {entry[1]}")
        print("End")
