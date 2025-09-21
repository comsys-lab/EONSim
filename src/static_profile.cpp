#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <cstdint>

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <dataset_path> <num_emb>" << std::endl;
        return 1;
    }

    std::string dataset_path = argv[1];
    uint64_t num_emb = std::stoull(argv[2]);

    // Get directory path and filename for output
    size_t last_slash = dataset_path.find_last_of("/");
    size_t last_dot = dataset_path.find_last_of(".");
    std::string dir_path = dataset_path.substr(0, last_slash + 1);
    std::string filename = dataset_path.substr(last_slash + 1, last_dot - last_slash - 1);
    
    // Read file and count occurrences in parallel
    std::vector<std::unordered_map<uint32_t, uint32_t>> thread_counts;
    std::ifstream file(dataset_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << dataset_path << std::endl;
        return 1;
    }

    // Count total lines first
    uint64_t total_lines = 0;
    std::string line;
    while (std::getline(file, line)) {
        total_lines++;
    }
    file.clear();
    file.seekg(0);

    std::cout << "Started counting " << total_lines << " entries..." << std::endl;

    // Initialize thread-local counters
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_counts.resize(num_threads);
        }
    }

    // Count occurrences in parallel
    uint64_t processed_lines = 0;
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        std::string line;
        uint32_t value;

        #pragma omp for schedule(dynamic) ordered
        for (uint64_t i = 0; i < total_lines; i++) {
            #pragma omp ordered
            {
                if (std::getline(file, line)) {
                    value = static_cast<uint32_t>(std::stoul(line));
                    thread_counts[thread_id][value]++;
                    
                    // Update progress
                    processed_lines++;
                    if (processed_lines % (total_lines/100) == 0) {
                        std::cout << "\rCounting progress: " 
                                << (processed_lines * 100 / total_lines) 
                                << "%" << std::flush;
                    }
                }
            }
        }
    }
    std::cout << "\rCounting complete!                 " << std::endl;

    std::cout << "Merging thread results..." << std::endl;
    // Merge counts from all threads
    std::unordered_map<uint32_t, uint32_t> total_counts;
    for (const auto& thread_map : thread_counts) {
        for (const auto& [key, count] : thread_map) {
            total_counts[key] += count;
        }
    }

    std::cout << "Sorting " << total_counts.size() << " unique entries..." << std::endl;
    // Convert to vector for sorting
    std::vector<std::pair<uint32_t, uint32_t>> sorted_counts(
        total_counts.begin(), total_counts.end()
    );

    // Sort by count in descending order
    std::sort(sorted_counts.begin(), sorted_counts.end(),
        [](const auto& a, const auto& b) {
            return a.second > b.second;
        }
    );
    
    std::cout << "Sorting complete!" << std::endl;
    
    // Display top 10 frequent numbers
    std::cout << "\nTop 10 most frequent numbers:" << std::endl;
    std::cout << "Rank\tNumber\tCount" << std::endl;
    std::cout << "------------------------" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), sorted_counts.size()); i++) {
        std::cout << i + 1 << "\t" 
                  << sorted_counts[i].first << "\t" 
                  << sorted_counts[i].second << std::endl;
    }
    std::cout << std::endl;
    
    // Create and open output file
    std::string output_path = dir_path + filename + "_profiled_" + std::to_string(num_emb) + ".txt";
    std::cout << "Writing results to " << output_path << "..." << std::endl;
    std::ofstream output_file(output_path);

    // Write sorted numbers up to num_emb lines
    uint64_t written_lines = 0;
    
    // Write actual numbers
    for (const auto& [number, count] : sorted_counts) {
        if (written_lines >= num_emb) break;
        output_file << number << "\n";
        written_lines++;
    }

    // Fill remaining lines with zeros if needed
    while (written_lines < num_emb) {
        output_file << "0\n";
        written_lines++;
    }

    std::cout << "Complete! Wrote " << written_lines << " entries." << std::endl;

    file.close();
    output_file.close();
    return 0;
}
