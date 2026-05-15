from cache_modules.lru_module import LRUModule


class LFUModule:
    """Set-local LFU state with LRU tie-breaking."""

    def __init__(self, cache_way, counter_bits=8, aging_interval=0):
        self.cache_way = cache_way
        self.counter_bits = counter_bits
        self.aging_interval = aging_interval
        self.max_counter = (1 << counter_bits) - 1

        self.freq = {}
        self.recency = LRUModule(cache_way)
        self.access_counter = 0

    def _apply_aging(self):
        for tag in list(self.freq.keys()):
            self.freq[tag] >>= 1

    def _tick(self):
        self.access_counter += 1
        if self.aging_interval > 0 and self.access_counter % self.aging_interval == 0:
            self._apply_aging()

    def access(self, tag):
        """Access a tag and update LFU/LRU state. Returns (hit, victim)."""
        self._tick()

        if tag in self.freq:
            self.freq[tag] = min(self.max_counter, self.freq[tag] + 1)
            self.recency.search_and_access(tag)
            return True, None

        victim = None
        if len(self.freq) >= self.cache_way:
            min_freq = min(self.freq.values())
            candidates = {line for line, value in self.freq.items() if value == min_freq}
            victim = self.recency.find_lru_in_candidates(candidates)
            if victim is None:
                victim = next(iter(candidates))

            self.recency.remove_addr(victim)
            del self.freq[victim]

        self.freq[tag] = 1
        self.recency.insert_node(tag)
        return False, None
