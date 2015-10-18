#ifndef PTXSYMBLTABLEH
#define PTXSYMBLTABLEH

#include "semantics/Variable.h"
#include <vector>
#include <mutex>

namespace ptx {
	union param_storage_t {
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
        unsigned long long ull;
	};

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
		bool has(const ptx::Variable& var) const;
		bool has(const std::string& name) const;
        bool getIfExists(const std::string& name, param_storage_t& storage) const;
		param_storage_t get(const ptx::Variable& var) const;
		param_storage_t get(const std::string& name) const;
		unsigned long long address(const std::string& name) const;
		ptx::Variable variable(const std::string& name) const;
		void print() const{
			for (const auto& x: _data)
                std::cout << x.var.name() << ": " << x.data.i << "\n";
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
		ProtectedStoragePtr _sharedData;
		SymbolStorage _data;
	public:
		void setSharedSection(ProtectedStoragePtr sharedData);
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
			return this->_sharedData->data.get(name);
		}
		unsigned long long address(const std::string& name) const {
			if (this->_sharedData->data.has(name))
				return this->_sharedData->data.address(name);
			return this->_data.address(name);
		}
		ptx::Variable variable(const std::string& name) const {
			if (this->_sharedData->data.has(name))
				return this->_sharedData->data.variable(name);
			return this->_data.variable(name);
		}
		void lockSharedSection(){
			this->_sharedData->mutex.lock();
		}
		void unlockSharedSection(){
			this->_sharedData->mutex.unlock();
		}
		void declareShared(const ptx::Variable& var) {
			this->lockSharedSection();
			if (!this->_sharedData->data.has(var)) {
				this->_sharedData->data.set(var, param_storage_t());
			}
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
