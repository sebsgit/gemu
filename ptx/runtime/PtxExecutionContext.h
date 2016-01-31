#ifndef PTXEXECCONTEXTH
#define PTXEXECCONTEXTH

class InstructionPtr;

#include "../arch/Device.h"
#include "cuda/cudaThreads.h"
#include "semantics/Semantics_fwd.h"
#include <algorithm>
#include <iostream>
#include "semantics/SymbolTable.h"

namespace ptx {

    #ifdef PTX_KERNEL_DEBUG
        namespace debug { class KernelDebugger; }
    #endif

	namespace exec {

        enum ThreadExecResult {
			ThreadExited,
			ThreadSuspended
		};

        enum BlockExecResult {
            BlockNotRunning,
            BlockRunning,
            BlockError,
            BlockOk
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
			void setProgramCounter(size_t pc);
			size_t programCounter() const;
			void exec(const InstructionList& list);
			void exec(const Instruction& i);
			void exec(const MemoryInstruction& i);
			void exec(const Return&);
			void exec(const FunctionDeclaration&);
			void exec(const ModuleDirective&);
			void exec(const VariableDeclaration&);
			void exec(const Branch& branch);
			void exec(const Barrier& barrier);
            void exec(const Call& call);
            ThreadExecResult result() const;
		private:
			gemu::cuda::Thread& _thread;
			SymbolTable& _symbols;
			const InstructionList * _instr = nullptr;
			size_t _pc = 0;
			bool _barrierWait=false;
    #ifdef PTX_KERNEL_DEBUG
        protected:
            static debug::KernelDebugger* _debugger;
            void exec_debug(const InstructionList& list);
            friend class debug::KernelDebugger;
    #endif
		};

		class PtxBlockDispatcher {
            struct Data;
		public:
			PtxBlockDispatcher(gemu::Device& device, gemu::cuda::ThreadBlock& block);
            ~PtxBlockDispatcher();
            void launch(ptx::Function& func, SymbolTable& symbols);
            void synchronize();
            BlockExecResult result() const { return this->_result; }
		private:
			gemu::Device& _device;
			gemu::cuda::ThreadBlock& _block;
            BlockExecResult _result = BlockExecResult::BlockNotRunning;
            Data* _data;
		};
	}
}

#endif
