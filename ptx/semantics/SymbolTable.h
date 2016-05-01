#ifndef PTXSYMBLTABLEH
#define PTXSYMBLTABLEH

#include "semantics/Variable.h"
#include <vector>
#include <cstring>
#include <memory>
#include <mutex>

namespace ptx {
	union param_raw_storage_t {
        unsigned long long data = 0;
        bool b;
        short s;
        int i;
        long l;
        long long ll;
        float f;
        double d;
		unsigned u;
        unsigned long ul;
	};

	class param_storage_t {
	public:
		/*implicit*/
		param_storage_t(const param_raw_storage_t& init=param_raw_storage_t())
			:_data(init)
		{}
		static param_storage_t allocArray(const size_t size) {
			param_storage_t result;
			result._allocAddr.reset(malloc(size), ::free);
			return result;
		}
	private:
		param_raw_storage_t _data;
		std::shared_ptr<void> _allocAddr;

		template<typename T> friend T& param_cast(param_storage_t&);
		template<typename T> friend T param_cast(const param_storage_t&);
		friend void param_copy_from(param_storage_t& output, const void* input, const size_t size);
		friend void param_copy_into(void* output, const param_storage_t& input, const size_t size);
	};

	using address_t = unsigned long long;

	template <typename T> T& param_cast(param_storage_t& s) { return -1; }
	template<> inline unsigned long long& param_cast(param_storage_t& s) { return s._data.data; }
	template<> inline bool& param_cast(param_storage_t& s) { return s._data.b; }
	template<> inline short& param_cast(param_storage_t& s) { return s._data.s; }
	template<> inline int& param_cast(param_storage_t& s) { return s._data.i; }
	template<> inline long& param_cast(param_storage_t& s) { return s._data.l; }
	template<> inline long long& param_cast(param_storage_t& s) { return s._data.ll; }
	template<> inline float& param_cast(param_storage_t& s) { return s._data.f; }
	template<> inline double& param_cast(param_storage_t& s) { return s._data.d; }
	template<> inline unsigned& param_cast(param_storage_t& s) { return s._data.u; }
	template<> inline unsigned long& param_cast(param_storage_t& s) { return s._data.ul; }
	template <typename T> T param_cast(const param_storage_t& s) { return -1; }
	template<> inline unsigned long long param_cast(const param_storage_t& s) { return s._data.data; }
	template<> inline bool param_cast(const param_storage_t& s) { return s._data.b; }
	template<> inline short param_cast(const param_storage_t& s) { return s._data.s; }
	template<> inline int param_cast(const param_storage_t& s) { return s._data.i; }
	template<> inline long param_cast(const param_storage_t& s) { return s._data.l; }
	template<> inline long long param_cast(const param_storage_t& s) { return s._data.ll; }
	template<> inline float param_cast(const param_storage_t& s) { return s._data.f; }
	template<> inline double param_cast(const param_storage_t& s) { return s._data.d; }
	template<> inline unsigned param_cast(const param_storage_t& s) { return s._data.u; }
	template<> inline unsigned long param_cast(const param_storage_t& s) { return s._data.ul; }
	inline void param_copy_from(param_storage_t& output, const void* input, const size_t size) {
		std::memcpy(output._allocAddr ? output._allocAddr.get() : &output._data.data, input, size);
	}
	inline void param_copy_into(void* output, const param_storage_t& input, const size_t size) {
		std::memcpy(output, input._allocAddr ? input._allocAddr.get() : &input._data.data, size);
	}

	class SymbolStorage {
	public:
		struct entry_t {
			ptx::Variable var;
			param_storage_t data;
			entry_t(const ptx::Variable& v=ptx::Variable(), const param_storage_t& d=param_storage_t())
			:var(v)
			,data(d)
			{}
		};
	public:
		void set(const ptx::Variable& var, const param_storage_t& storage);
		void set(const std::string& name, const param_storage_t& storage);
        bool setIfExists(const ptx::Variable& var, const param_storage_t& storage);
        bool setIfExists(const std::string& name, const param_storage_t& storage);
        void declare(const ptx::Variable& var);
        bool getIfExists(const std::string& name, param_storage_t& storage) const;
        bool getIfExists(const std::string& name, ptx::Variable& var) const;
		param_storage_t get(const ptx::Variable& var) const;
		param_storage_t get(const std::string& name) const;
		unsigned long long address(const std::string& name) const;
		void print() const{
			for (const auto& x: _data)
				std::cout << x.var.name() << ": " << param_cast<int>(x.data) << "\n";
		}

	private:
		std::vector<entry_t> _data;
	};

	typedef struct {
		SymbolStorage data;
		std::mutex mutex;
	} ProtectedStorage;

	typedef std::shared_ptr<ProtectedStorage> ProtectedStoragePtr;

	class SymbolTable {
	private:
        std::shared_ptr<ptx::SymbolStorage> _globalData;
		ProtectedStoragePtr _sharedData;
		SymbolStorage _data;
	public:
		void setSharedSection(ProtectedStoragePtr sharedData);
        void setGlobalSection(std::shared_ptr<ptx::SymbolStorage> data){ this->_globalData = data; }
		ProtectedStoragePtr sharedSection() const {return this->_sharedData;}
		void set(const ptx::Variable& var, const param_storage_t& storage){
			if (var.space() == AllocSpace::Shared) {
				this->_sharedData->data.set(var, storage);
			} else {
				this->_data.set(var, storage);
			}
		}
		void set(const std::string& name, const param_storage_t& storage) {
            if (this->_data.setIfExists(name, storage) == false)
                if (this->_globalData->setIfExists(name, storage) == false)
                    this->_sharedData->data.setIfExists(name, storage);
		}
		param_storage_t get(const ptx::Variable& var) const {
			if (var.space() == AllocSpace::Shared)
				return this->_sharedData->data.get(var);
			else
				return this->_data.get(var);
		}
		param_storage_t get(const std::string& name) const {
            param_storage_t tmp;
            if (this->_data.getIfExists(name, tmp))
                return tmp;
            else if (this->_globalData->getIfExists(name, tmp))
                return tmp;
			return this->_sharedData->data.get(name);
		}
        param_storage_t operator[](const std::string& name) const {
            return this->get(name);
        }
		unsigned long long address(const std::string& name) const {
            unsigned long long result = this->_data.address(name);
            if (result == 0)
                result = this->_sharedData->data.address(name);
            return result;
		}
		ptx::Variable variable(const std::string& name) const {
            ptx::Variable result;
            if (this->_data.getIfExists(name, result) == false)
                this->_sharedData->data.getIfExists(name, result);
            return result;
		}
		void lockSharedSection(){
			this->_sharedData->mutex.lock();
		}
		void unlockSharedSection(){
			this->_sharedData->mutex.unlock();
		}
		void declareShared(const ptx::Variable& var) {
			this->lockSharedSection();
            this->_sharedData->data.declare(var);
			this->unlockSharedSection();
		}
        void declare(const ptx::Variable& var) {
            this->_data.declare(var);
        }
        void print() const {
            this->_data.print();
        }
	};

}

#endif
