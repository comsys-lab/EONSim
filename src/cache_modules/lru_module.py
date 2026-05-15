import numpy as np

class Node:
    """Doubly-linked list node for LRU cache"""
    def __init__(self, addr=None):
        self.addr = addr
        self.prev = None
        self.next = None

class LRUModule:
    """Hash map + doubly-linked list based LRU cache for O(1) operations"""
    def __init__(self, cache_way):
        self.cache_way = cache_way
        self.cache_map = {}  # addr -> Node mapping for O(1) lookup
        
        # Dummy head and tail nodes to simplify edge cases
        self.head = Node()
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head
        
        self.size = 0

    def get_cache_way(self):
        return self.cache_way

    def _remove_node(self, node):
        """Remove node from doubly-linked list"""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def _add_to_front(self, node):
        """Add node right after head (most recently used position)"""
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def insert_node(self, addr):
        """Insert new address into cache"""
        if addr in self.cache_map:
            # Already exists, just move to front
            node = self.cache_map[addr]
            self._remove_node(node)
            self._add_to_front(node)
            return
        
        # Create new node
        new_node = Node(addr)
        self.cache_map[addr] = new_node
        self._add_to_front(new_node)
        self.size += 1
        
        if self.size > self.cache_way:
            lru_node = self.tail.prev
            self._remove_node(lru_node)
            del self.cache_map[lru_node.addr]
            self.size -= 1

    def search_and_access(self, addr):
        """Search for address and move to front if found. Returns True on hit."""
        if addr not in self.cache_map:
            return False
        
        # Cache hit: move to front (most recently used)
        node = self.cache_map[addr]
        self._remove_node(node)
        self._add_to_front(node)
        return True

    def remove_addr(self, addr): # for LFU
        """Remove a specific address from cache if present."""
        if addr not in self.cache_map:
            return False

        node = self.cache_map[addr]
        self._remove_node(node)
        del self.cache_map[addr]
        self.size -= 1
        return True

    def find_lru_in_candidates(self, candidates): # for LFU tie-breaking
        """Return the least-recently-used address among candidates."""
        current = self.tail.prev
        while current != self.head:
            if current.addr in candidates:
                return current.addr
            current = current.prev
        return None

    def is_empty(self):
        """Check if cache is empty"""
        return self.size == 0

    def return_as_array(self):
        """Return cache contents as numpy array (MRU to LRU order)"""
        addr_list = []
        current = self.head.next
        
        # Traverse from head to tail (MRU to LRU)
        while current != self.tail and len(addr_list) < self.cache_way:
            addr_list.append(current.addr if current.addr is not None else 0)
            current = current.next
        
        # Pad with zeros if needed
        while len(addr_list) < self.cache_way:
            addr_list.append(0)
        
        return np.array(addr_list, dtype=np.int64)

    def print_list(self):
        """Print cache contents for debugging (MRU to LRU)"""
        current = self.head.next
        while current != self.tail:
            print(current.addr, end=" -> ")
            current = current.next
        print("End")