#ifndef __MEMORY_SYSTEM__H
#define __MEMORY_SYSTEM__H

#include <memory> // added for std::unique_ptr
#include <cstdint> // added for uint64_t
#include <functional>
#include <string>

namespace dramsim3 {

// This should be the interface class that deals with CPU
class MemorySystem {
   public:
    MemorySystem(const std::string &config_file, const std::string &output_dir,
                 std::function<void(uint64_t, int)> read_callback,
                 std::function<void(uint64_t, int)> write_callback,
				 int module_idx);
    ~MemorySystem();
    void ClockTick();
    void RegisterCallbacks(std::function<void(uint64_t, int)> read_callback,
                           std::function<void(uint64_t, int)> write_callback);
    double GetTCK() const;
    int GetBusBits() const;
    int GetBurstLength() const;
    int GetQueueSize() const;
    void PrintStats() const;
    void ResetStats();

    bool WillAcceptTransaction(uint64_t hex_addr, bool is_write) const;
    bool AddTransaction(uint64_t hex_addr, bool is_write);

	int getChannel(uint64_t hex_addr);

   private:
	void *config_;
	void* dram_system;
	int module_index;
};

MemorySystem* GetMemorySystem(const std::string &config_file, const std::string &output_dir,
                 std::function<void(uint64_t, int)> read_callback,
                 std::function<void(uint64_t, int)> write_callback,
				 int module_idx);
}  // namespace dramsim3

#endif
