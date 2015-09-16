#ifndef GEMUDEVICEH
#define GEMUDEVICEH

#include <memory>
#include <vector>
#include <unordered_map>

namespace gemu {
	class MemoryBlock {
		struct free_block_t {
			size_t start;
			size_t size;
			free_block_t (size_t st=0, size_t sz=0) : start(st), size(sz) {

			}
		};
	public:
		MemoryBlock(size_t size)
			:_data(new char[size])
			,_size(size)
		{
			this->_freeBlocks.push_back(free_block_t(0, size));
		}
		~MemoryBlock(){
			delete [] _data;
		}
		size_t size() const {
			return this->_size;
		}
		void * alloc(size_t size) {
			for (auto& block : this->_freeBlocks) {
				if (block.size < size) {
					const size_t start = block.start;
					block.start += size;
					block.size -= size;
					this->_allocData[this->_data + start] = size;
					return this->_data + start;
				}
			}
			return nullptr;
		}
		bool isValid(void * ptr) const {
			return this->_allocData.find(ptr) != this ->_allocData.end();
		}
		bool free(void * ptr) {
			auto it = this->_allocData.find(ptr);
			if (it != this->_allocData.end()) {
				this->_freeBlocks.push_back(free_block_t((char*)ptr - this->_data, it->second));
				this->_allocData.erase(it);
				//TODO perform merge when too many free blocks to avoid fragmentation
				return true;
			}
			return false;
		}
	private:
		MemoryBlock(const MemoryBlock&);
		MemoryBlock& operator=(const MemoryBlock&);
	private:
		char * _data = nullptr;
		size_t _size = 0;
		std::vector<free_block_t> _freeBlocks;
		std::unordered_map<void*, size_t> _allocData;
	};
	typedef std::shared_ptr<MemoryBlock> MemoryBlockPtr;

	class Device {
	public:
		Device (size_t memorySize)
			:_memory(std::make_shared<MemoryBlock>(memorySize))
		{
		}
		~Device() {
		}
		MemoryBlockPtr memory() {
			return this->_memory;
		}
	private:
		MemoryBlockPtr _memory;
	};

	class AbstractExecutionContext {
	public:
		AbstractExecutionContext(Device& device)
		:_device(device)
		{}
		virtual ~AbstractExecutionContext(){}
	protected:
		Device& _device;
	};

}

#endif
