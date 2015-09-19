#ifndef GEMUCUDADRIVERAPIH
#define GEMUCUDADRIVERAPIH

#include "cudaDefines.h"
#include "cudaForward.h"
#include "arch/Device.h"
#include "semantics/Function.h"
#include <cstring>
#include <vector>
#include <algorithm>

namespace gemu {
	namespace cuda {
		class Module {
		public:
			CUmodule id() const {
				return this->_id;
			}
			void add(const ptx::Function& func) {
				this->_functions.push_back(func);
			}
		private:
			CUmodule _id;
			std::vector<ptx::Function> _functions;
			friend class GlobalContext;
		};
		class GlobalContext {
		public:
			CUmodule add(const Module& m) {
				Module toAdd = m;
				toAdd._id = (CUmodule)++this->_nextModuleId;
				this->_modules.push_back(toAdd);
				return toAdd._id;
			}
			bool remove(CUmodule modId) {
				auto it = std::find_if(this->_modules.begin(),
									   this->_modules.end(),
								   		[&](const Module& m) {return m.id() == modId;});
				if (it != this->_modules.end()) {
					this->_modules.erase(it);
					return true;
				}
				return false;
			}
			CUfunction function(CUmodule modId, const char * name) {
				auto it = std::find_if(this->_modules.begin(),
									   this->_modules.end(),
								   		[&](const Module& m) {return m.id() == modId;});
				if (it != this->_modules.end()) {
					for (size_t i=0 ; i<it->_functions.size() ; ++i) {
						if (it->_functions[i].name() == std::string(name)) {
							auto result = reinterpret_cast<CUfunction>(reinterpret_cast<unsigned long>(modId) * 1000 + i);
							this->_funcCache[result] = it->_functions[i];
							return result;
						}
					}
				}
				return reinterpret_cast<CUfunction>(-1);
			}
			ptx::Function function(CUfunction fhandle) const {
				return this->_funcCache.find(fhandle) != this->_funcCache.end() ? this->_funcCache.find(fhandle)->second : ptx::Function();
			}
		private:
			std::vector<Module> _modules;
			std::unordered_map<CUfunction, ptx::Function> _funcCache;
			unsigned long long _nextModuleId = 0;
		};
	}
}

extern gemu::cuda::GlobalContext * _driverContext;
extern gemu::Device * _default_cuda_device;
extern const CUdevice _default_cuda_device_id;

#endif
