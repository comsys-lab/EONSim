#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <algorithm>
#include <utility>
#include <cstdint>

namespace py = pybind11;

class SRRIPCache {
private:
    // Core data structures
    std::vector<std::vector<std::pair<int64_t, uint8_t>>> cache_sets;
    size_t cache_way;
    uint8_t rrpv_bits;
    uint8_t rrpv_insert;
    uint8_t max_rrpv;
    size_t num_sets;
    size_t offset_bits;

    size_t get_set_index(int64_t addr) const {
        // Fully associative cache for now
        return 0;
    }

public:
    SRRIPCache(size_t way, uint8_t bits, uint8_t insert_val) 
        : cache_way(way)
        , rrpv_bits(bits)
        , rrpv_insert(insert_val)
        , max_rrpv((1 << bits) - 1)
        , num_sets(1)  // Fully associative
        , offset_bits(6) {  // 64-byte cache line
        cache_sets.resize(num_sets);
    }

    bool access(int64_t tag) {
        size_t set_idx = get_set_index(tag);
        auto& cache_set = cache_sets[set_idx];
        
        // Look for tag in cache
        auto it = std::find_if(cache_set.begin(), cache_set.end(),
            [tag](const std::pair<int64_t, uint8_t>& entry) { 
                return entry.first == tag; 
            });

        if (it != cache_set.end()) {
            // Cache hit: Reset RRPV to 0
            it->second = 0;
            return true;
        }

        // Cache miss handling
        if (cache_set.size() < cache_way) {
            cache_set.emplace_back(tag, rrpv_insert);
        } else {
            bool replaced = false;
            while (!replaced) {
                // Find entry with max RRPV
                auto max_rrpv_it = std::find_if(cache_set.begin(), cache_set.end(),
                    [this](const std::pair<int64_t, uint8_t>& entry) {
                        return entry.second >= max_rrpv;
                    });

                if (max_rrpv_it != cache_set.end()) {
                    // Replace the victim
                    *max_rrpv_it = std::make_pair(tag, rrpv_insert);
                    replaced = true;
                } else {
                    // Age all entries
                    for (auto& entry : cache_set) {
                        if (entry.second < max_rrpv) {
                            entry.second++;
                        }
                    }
                }
            }
        }
        return false;
    }

    py::array_t<int64_t> get_entries() {
        std::vector<int64_t> result;
        result.reserve(cache_way);
        
        for (const auto& entry : cache_sets[0]) {
            result.push_back(entry.first);
        }
        
        // Zero padding
        result.resize(cache_way, 0);
        
        return py::array_t<int64_t>(
            {static_cast<int64_t>(cache_way)},
            result.data()
        );
    }

    bool is_empty() {
        return cache_sets[0].empty();
    }

    // Return entries for a specific set with their RRPV values
    py::array_t<int64_t> get_set_entries(size_t set_idx) {
        auto& cache_set = cache_sets[set_idx];
        
        // Create a vector to hold our flattened data
        std::vector<int64_t> flat_data(cache_way * 2, 0);  // Initialize with zeros
        
        // Fill with actual entries
        for (size_t i = 0; i < cache_set.size(); ++i) {
            flat_data[i*2] = cache_set[i].first;     // tag
            flat_data[i*2 + 1] = cache_set[i].second;  // RRPV
        }
        
        // Create the array with shape information
        std::vector<ssize_t> shape = {static_cast<ssize_t>(cache_way), 2};
        std::vector<ssize_t> strides = {2 * sizeof(int64_t), sizeof(int64_t)};
        
        return py::array_t<int64_t>(
            shape,
            strides,
            flat_data.data()
        );
    }

    size_t get_num_entries(size_t set_idx) {
        return cache_sets[set_idx].size();
    }

    size_t get_num_sets() const {
        return num_sets;
    }
};

PYBIND11_MODULE(srrip_cache, m) {
    py::class_<SRRIPCache>(m, "SRRIPCache")
        .def(py::init<size_t, uint8_t, uint8_t>())
        .def("access", &SRRIPCache::access)
        .def("get_entries", &SRRIPCache::get_entries)
        .def("is_empty", &SRRIPCache::is_empty)
        .def("get_set_entries", &SRRIPCache::get_set_entries)
        .def("get_num_entries", &SRRIPCache::get_num_entries)
        .def("get_num_sets", &SRRIPCache::get_num_sets);
}
