#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <omp.h>
#include <cstdint>
#include <cmath>
#include <filesystem>

int main(int argc, char* argv[]) {
    // Check arguments
    if (argc != 11) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dataset_path> <num_emb> <num_batch> <num_table> <batch_sz> "
                  << "<lookup_per_table> <emb_dim> <mem_gran> <n_format> <rows_per_table>" 
                  << std::endl;
        return 1;
    }

    // Parse arguments
    std::string dataset_path = argv[1];
    uint64_t num_emb = std::stoull(argv[2]);      // Reserved for future use
    uint64_t num_batch = std::stoull(argv[3]);
    uint64_t num_table = std::stoull(argv[4]);
    uint64_t batch_sz = std::stoull(argv[5]);
    uint64_t lookup_per_table = std::stoull(argv[6]);
    uint64_t emb_dim = std::stoull(argv[7]);
    uint64_t mem_gran = std::stoull(argv[8]);
    uint64_t n_format = std::stoull(argv[9]);
    uint64_t rows_per_table = std::stoull(argv[10]);

    // Print all arguments
    std::cout << "\n[DEBUG] Arguments:" << std::endl;
    std::cout << "dataset_path: " << dataset_path << std::endl;
    std::cout << "num_emb: " << num_emb 
              << ", num_batch: " << num_batch 
              << ", num_table: " << num_table 
              << ", batch_sz: " << batch_sz 
              << ", lookup_per_table: " << lookup_per_table 
              << ", emb_dim: " << emb_dim 
              << ", mem_gran: " << mem_gran 
              << ", n_format: " << n_format 
              << ", rows_per_table: " << rows_per_table 
              << std::endl << std::endl;

    // Get input filename without path and extension
    size_t last_slash = dataset_path.find_last_of("/");
    std::string full_filename = dataset_path.substr(last_slash + 1);
    size_t last_dot = full_filename.find_last_of(".");
    std::string filename = full_filename.substr(0, last_dot);

    // Create output directory path
    std::string output_dir = "datasets/profiled_datasets/";
    
    // Create directories if they don't exist using filesystem
    try {
        std::filesystem::create_directories(output_dir);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error creating directory: " << e.what() << std::endl;
        return 1;
    }

    // Read all numbers from file into a vector first
    std::vector<uint32_t> numbers;
    std::ifstream file(dataset_path);
    
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << dataset_path << std::endl;
        return 1;
    }

    std::string line;
    while (std::getline(file, line)) {
        numbers.push_back(static_cast<uint32_t>(std::stoul(line)));
    }
    
    if (numbers.empty()) {
        std::cerr << "No numbers found in the file!" << std::endl;
        return 1;
    }

    // Allocate 3D array
    std::vector<std::vector<std::vector<uint32_t>>> index_array(
        num_batch,
        std::vector<std::vector<uint32_t>>(
            num_table,
            std::vector<uint32_t>(batch_sz * lookup_per_table)
        )
    );

    // Fill the 3D array with numbers (cycling through input if needed)
    size_t number_idx = 0;
    const size_t total_numbers = numbers.size();
    
    std::cout << "Filling 3D array with dimensions [" 
              << num_batch << "][" << num_table << "][" 
              << batch_sz * lookup_per_table << "]..." << std::endl;

    for (uint64_t i = 0; i < num_batch; i++) {
        for (uint64_t j = 0; j < num_table; j++) {
            for (uint64_t k = 0; k < batch_sz * lookup_per_table; k++) {
                index_array[i][j][k] = numbers[number_idx];
                number_idx = (number_idx + 1) % total_numbers; // Cycle through numbers
            }
        }
    }

    std::cout << "3D array filled successfully." << std::endl;


    // [DEBUG]
    // Save index_array to file for verification
    // std::string index_output_path = output_dir + filename + "_index_array_" + 
    //                                std::to_string(num_batch) + "_" +
    //                                std::to_string(num_table) + "_" +
    //                                std::to_string(batch_sz) + "_" +
    //                                std::to_string(lookup_per_table) + ".txt";
    
    // std::cout << "Writing index array to " << index_output_path << "..." << std::endl;
    // std::ofstream index_file(index_output_path);

    // // Write indices to file
    // for (const auto& batch : index_array) {
    //     for (const auto& table : batch) {
    //         for (const auto& idx : table) {
    //             index_file << idx << "\n";
    //         }
    //     }
    // }
    // index_file.close();
    
    // std::cout << "Index array file created successfully." << std::endl;

    std::cout << "Converting indices to virtual addresses..." << std::endl;

    // Calculate access_per_vector like Python
    uint64_t access_per_vector = std::ceil(static_cast<double>(emb_dim * n_format) / mem_gran);
    
    // Create addr_array with dimensions matching Python's addr_trace shape
    std::vector<std::vector<std::vector<uint64_t>>> addr_array(
        num_batch,
        std::vector<std::vector<uint64_t>>(
            num_table,
            std::vector<uint64_t>(batch_sz * lookup_per_table * access_per_vector)
        )
    );

    std::cout << "[DEBUG] addr_array shape: [" << num_batch << "][" << num_table << "][" 
              << batch_sz * lookup_per_table * access_per_vector << "]" << std::endl;

    // Convert indices to memory addresses exactly as in Python
    for (uint64_t nb = 0; nb < num_batch; nb++) {
        std::cout << "Converting vector indices into virtual memory addresses for batch " << nb << "..." << std::endl;
        for (uint64_t nt = 0; nt < num_table; nt++) {
            for (uint64_t vec = 0; vec < batch_sz * lookup_per_table; vec++) {
                for (uint64_t dim = 0; dim < access_per_vector; dim++) {
                    // Match Python's bit_length() calculation
                    uint64_t val = emb_dim * n_format - 1;
                    uint64_t bytes_per_vec = 0;
                    while (val > 0) {
                        bytes_per_vec++;
                        val >>= 1;
                    }
                    
                    // Debug print for verification
                    if (nb == 0 && nt == 0 && vec == 0 && dim == 0) {
                        std::cout << "[DEBUG] First address calculation:" << std::endl;
                        std::cout << "bytes_per_vec: " << bytes_per_vec << std::endl;
                        std::cout << "log2(rows_per_table): " << std::log2(rows_per_table) << std::endl;
                        std::cout << "index value: " << index_array[nb][nt][vec] << std::endl;
                    }

                    uint64_t tbl_bits = nt;
                    tbl_bits <<= (static_cast<uint64_t>(std::log2(rows_per_table)) + bytes_per_vec);
                    
                    uint64_t vec_idx = static_cast<uint64_t>(index_array[nb][nt][vec]);
                    vec_idx <<= bytes_per_vec;
                    
                    uint64_t dim_bits = mem_gran * dim;
                    uint64_t this_addr = tbl_bits + vec_idx + dim_bits;

                    // Debug first few addresses
                    if (nb == 0 && nt == 0 && vec == 0 && dim < 5) {
                        std::cout << "[DEBUG] Address " << dim << ": 0x" 
                                << std::hex << this_addr << std::dec << std::endl;
                    }

                    addr_array[nb][nt][vec * access_per_vector + dim] = this_addr;
                }
            }
        }
    }

    std::cout << "Address conversion complete!" << std::endl;
    std::cout << "Counting address occurrences..." << std::endl;

    // Initialize thread-local counters for parallel counting
    std::vector<std::unordered_map<uint64_t, uint32_t>> thread_counts;
    int num_threads;
    #pragma omp parallel
    {
        #pragma omp single
        {
            num_threads = omp_get_num_threads();
            thread_counts.resize(num_threads);
        }
    }

    // Count address occurrences in parallel
    uint64_t total_addrs = num_batch * num_table * addr_array[0][0].size();
    uint64_t processed_addrs = 0;

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        
        #pragma omp for schedule(dynamic)
        for (uint64_t nb = 0; nb < num_batch; nb++) {
            for (uint64_t nt = 0; nt < num_table; nt++) {
                for (const auto& addr : addr_array[nb][nt]) {
                    thread_counts[thread_id][addr]++;
                    
                    #pragma omp atomic
                    processed_addrs++;
                    
                    if (processed_addrs % (total_addrs/100) == 0) {
                        #pragma omp critical
                        {
                            std::cout << "\rCounting progress: " 
                                     << (processed_addrs * 100 / total_addrs) 
                                     << "%" << std::flush;
                        }
                    }
                }
            }
        }
    }
    std::cout << "\rCounting complete!" << std::endl;

    // Merge thread counts
    std::cout << "Merging thread results..." << std::endl;
    std::unordered_map<uint64_t, uint32_t> total_counts;
    for (const auto& thread_map : thread_counts) {
        for (const auto& [addr, count] : thread_map) {
            total_counts[addr] += count;
        }
    }

    // Convert to vector for sorting
    std::cout << "Sorting " << total_counts.size() << " unique addresses..." << std::endl;
    std::vector<std::pair<uint64_t, uint32_t>> sorted_counts(
        total_counts.begin(), total_counts.end()
    );

    // Parallel sort by count in descending order
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::sort(sorted_counts.begin(), sorted_counts.end(),
                [](const auto& a, const auto& b) {
                    return a.second > b.second;
                }
            );
        }
    }

    // Display top 10 frequent addresses
    std::cout << "\nTop 10 most frequent addresses:" << std::endl;
    std::cout << "Rank\tAddress\t\tCount" << std::endl;
    std::cout << "--------------------------------" << std::endl;
    for (size_t i = 0; i < std::min(size_t(10), sorted_counts.size()); i++) {
        std::cout << i + 1 << "\t0x" 
                  << std::dec << sorted_counts[i].first << std::dec << "\t" 
                  << sorted_counts[i].second << std::endl;
    }
    std::cout << std::endl;

    // Create output file with profiling results
    std::string profile_output_path = output_dir + filename + "_" + std::to_string(num_emb) + ".txt";
    
    std::cout << "Writing profiling results to " << profile_output_path << "..." << std::endl;
    std::ofstream profile_file(profile_output_path);

    // Write top num_emb addresses
    uint64_t written_lines = 0;
    for (const auto& [addr, count] : sorted_counts) {
        if (written_lines >= num_emb) break;
        profile_file << addr << "\n";
        written_lines++;
    }

    // Fill remaining lines with zeros if needed
    while (written_lines < num_emb) {
        profile_file << "0\n";
        written_lines++;
    }

    std::cout << "Complete! Wrote " << written_lines << " entries." << std::endl;

    // [DEBUG]
    // Create and open output file for addresses
    // std::string addr_output_path = output_dir + filename + "_addr_" + 
    //                               std::to_string(num_batch) + "_" +
    //                               std::to_string(num_table) + "_" +
    //                               std::to_string(batch_sz) + "_" +
    //                               std::to_string(lookup_per_table) + ".txt";
    
    // std::cout << "Writing addresses to " << addr_output_path << "..." << std::endl;
    // std::ofstream addr_file(addr_output_path);

    // // Write addresses to file
    // for (const auto& batch : addr_array) {
    //     for (const auto& table : batch) {
    //         for (const auto& addr : table) {
    //             addr_file << addr << "\n";
    //         }
    //     }
    // }
    // addr_file.close();

    // std::cout << "Complete! Address file created successfully." << std::endl;

    file.close();
    // addr_file.close();
    // index_file.close();
    profile_file.close();
    return 0;
}
