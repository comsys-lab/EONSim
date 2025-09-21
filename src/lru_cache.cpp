#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <unordered_map>
#include <list>
#include <vector>

namespace py = pybind11;

class LRUCache {
private:
    size_t capacity;
    std::list<int64_t> lru_list;
    std::unordered_map<int64_t, std::list<int64_t>::iterator> cache_map;

public:
    LRUCache(size_t size) : capacity(size) {}

    bool search_and_access(int64_t key) {
        auto it = cache_map.find(key);
        if (it != cache_map.end()) {
            // Cache hit: move to front
            lru_list.erase(it->second);
            lru_list.push_front(key);
            it->second = lru_list.begin();
            return true;
        }
        return false;
    }

    void insert_node(int64_t key) {
        if (cache_map.size() >= capacity) {
            // Remove least recently used
            int64_t lru_key = lru_list.back();
            cache_map.erase(lru_key);
            lru_list.pop_back();
        }
        
        // Insert new key
        lru_list.push_front(key);
        cache_map[key] = lru_list.begin();
    }

    py::array_t<int64_t> return_as_array() {
        std::vector<int64_t> result;
        result.reserve(capacity);
        for (const auto& key : lru_list) {
            result.push_back(key);
        }
        while (result.size() < capacity) {
            result.push_back(0);
        }
        
        return py::array_t<int64_t>(
            {static_cast<int64_t>(capacity)},
            result.data()
        );
    }

    bool is_empty() {
        return lru_list.empty();
    }
};

PYBIND11_MODULE(lru_cache, m) {
    py::class_<LRUCache>(m, "LRUCache")
        .def(py::init<size_t>())
        .def("search_and_access", &LRUCache::search_and_access)
        .def("insert_node", &LRUCache::insert_node)
        .def("return_as_array", &LRUCache::return_as_array)
        .def("is_empty", &LRUCache::is_empty);
}
