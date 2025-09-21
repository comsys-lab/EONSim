import numpy as np

class Node: # You can think a node as a "cache line", and each cache line contains a memory address.
    def __init__(self, addr=None):
        self.addr = addr  # memory address
        self.next = None  # Next node pointer (if there is no next node, it becomes None)

class LRUlist:
    def __init__(self, cache_way):
        self.head = Node(None)
        self.cache_way = cache_way # Length of the linked list == cache_way
        
        # Init the linked list with null (number of nodes==cache_way).
        current = self.head
        for _ in range(1, cache_way):
            current.next = Node(None)
            current = current.next

    def get_cache_way(self):
        return self.cache_way

    def insert_node(self, value):
        new_node = Node(value)
        if not self.head:  # if list is empty
            self.head = new_node
            return

        # if the list already has a head node.
        new_node.next = self.head
        self.head = new_node

        current = self.head
        prev = None
        count = 0
        while current and count < self.cache_way:
            prev = current
            current = current.next
            count += 1
        
        if prev and prev.next:
            prev.next = None

    def search_and_access(self, addr_to_find):
        # Find the node -> if hit: that node becomes a new head, if miss: replacement.
        current = self.head
        prev = None

        # Searching the node
        while current:
            if current.addr == addr_to_find: # FOUND=Cache hit!
                if prev:  # if not current==head (if node is a head, then prev==None)
                    prev.next = current.next  # remove the node from the list
                    current.next = self.head  # The node becomes a new head
                    self.head = current # The node becomes a new head
                return True
            prev = current
            current = current.next
            
        return False # Cache miss

    def return_as_array(self):
        # Preallocate list with known size for better performance
        addr_list = [0] * self.cache_way
        current = self.head
        idx = 0
        
        # Fill the list with addresses
        while current and idx < self.cache_way:
            addr_list[idx] = current.addr if current.addr is not None else 0
            current = current.next
            idx += 1
            
        # Convert to numpy array with int64 dtype
        return np.array(addr_list, dtype=np.int64)

    def print_list(self):
        # This method is for debugging.
        current = self.head
        while current:
            print(current.addr, end=" -> ")
            current = current.next
        print("End")