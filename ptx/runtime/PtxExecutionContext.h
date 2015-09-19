#ifndef PTXEXECCONTEXTH
#define PTXEXECCONTEXTH

class InstructionPtr;

#include "../arch/Device.h"
#include "cuda/cudaThreads.h"
#include "semantics/Semantics_fwd.h"
#include "semantics/Variable.h"
#include <algorithm>

namespace ptx {
	namespace exec {

		struct param_storage_t {
			unsigned long long data = 0;
		};

		class SymbolTable {
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
			bool has(const ptx::Variable& var) const;
			bool has(const std::string& name) const;
			param_storage_t get(const ptx::Variable& var) const;
			param_storage_t get(const std::string& name) const;
		private:
			std::vector<entry_t> _data;
		};

		class PtxExecutionContext : public gemu::AbstractExecutionContext {
		public:
			PtxExecutionContext (gemu::Device& device,
								 gemu::cuda::Thread& thread,
							 	 SymbolTable& symbols)
				:gemu::AbstractExecutionContext(device)
				,_thread(thread)
				,_symbols(symbols)
			{}
			~PtxExecutionContext(){}
			void exec(const InstructionList& list);
			void exec(const Instruction& i);
			void exec(const Load& load);
			void exec(const Store& store);
			void exec(const Move& move);
			void exec(const Return&);
			void exec(const Convert&);
			void exec(const FunctionDeclaration&);
			void exec(const ModuleDirective&);
			void exec(const VariableDeclaration&);
		private:
			gemu::cuda::Thread& _thread;
			SymbolTable& _symbols;
		};

		class PtxBlockDispatcher {
		public:
			PtxBlockDispatcher(gemu::Device& device, gemu::cuda::ThreadBlock& block);
			bool launch(ptx::Function& func, SymbolTable& symbols);
		private:
			gemu::Device& _device;
			gemu::cuda::ThreadBlock& _block;
		};
	}
}

#endif
