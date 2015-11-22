#ifndef GEMUCUDADRIVERAPIH
#define GEMUCUDADRIVERAPIH

#include "cudaDefines.h"
#include "cudaStream.h"
#include "drivers/cuda/cuda.h"
#include "arch/Device.h"
#include "semantics/Function.h"
#include <cstring>
#include <vector>
#include <algorithm>

extern gemu::cuda::Stream * _default_cuda_stream;
extern gemu::Device * _default_cuda_device;
extern const CUdevice _default_cuda_device_id;

namespace gemu {
	namespace cuda {
		class Module {
		public:
			Module():_id(nullptr) {}
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
            ptx::Function function(const std::string& name) const {
                for (const auto& m : this->_modules) {
                    for (const auto& f : m._functions){
                        if (f.name() == name)
                            return f;
                    }
                }
                return ptx::Function();
            }
            CUstream createStream(unsigned int flags=CU_STREAM_DEFAULT){
                Stream * stream = new Stream(*_default_cuda_device, flags);
                this->_streams[(CUstream)stream] = stream;
                return (CUstream)stream;
            }
            bool destroyStream(CUstream stream){
                auto it = this->_streams.find(stream);
                if (it != this->_streams.end()){
                    delete it->second;
                    this->_streams.erase(stream);
                    return true;
                }
                return false;
            }
            bool findStream(CUstream streamId, Stream** stream) {
                if (streamId==0){
                    *stream = _default_cuda_stream;
                    return true;
                }
                auto it = this->_streams.find(streamId);
                if (it != this->_streams.end()){
                    *stream = it->second;
                    return true;
                }
                return false;
            }
		private:
			std::vector<Module> _modules;
			std::unordered_map<CUfunction, ptx::Function> _funcCache;
            std::unordered_map<CUstream, Stream*> _streams;
			unsigned long long _nextModuleId = 0;

            friend class PtxExecutionContext;
		};
	}
}

extern gemu::cuda::GlobalContext * _driverContext;

#endif
