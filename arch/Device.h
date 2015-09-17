#ifndef GEMUDEVICEH
#define GEMUDEVICEH

#include <memory>
#include <vector>
#include <unordered_map>

namespace gemu {
	class MemoryBlock {
	public:
		MemoryBlock(size_t size)
			:_maximumSize(size)
		{}
		size_t currentSize() const {
			return this->_size;
		}
		size_t maximumSize() const {
			return this->_maximumSize;
		}
		size_t freeSpace() const {
			return this->maximumSize() - this->currentSize();
		}
		void * alloc(size_t size) {
			if (this->_size + size <= this->_maximumSize){
				this->_size += size;
				void * result = malloc(size);
				this->_allocData[result] = size;
				return result;
			}
			return nullptr;
		}
		bool isValid(void * ptr) const {
			return this->_allocData.find(ptr) != this ->_allocData.end();
		}
		bool free(void * ptr) {
			auto it = this->_allocData.find(ptr);
			if (it != this->_allocData.end()) {
				this->_size -= it->second;
				this->_allocData.erase(it);
				free(ptr);
				return true;
			}
			return false;
		}
	private:
		MemoryBlock(const MemoryBlock&);
		MemoryBlock& operator=(const MemoryBlock&);
	private:
		const size_t _maximumSize;
		size_t _size = 0;
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
