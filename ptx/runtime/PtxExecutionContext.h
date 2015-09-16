#ifndef PTXEXECCONTEXTH
#define PTXEXECCONTEXTH

class InstructionPtr;

#include "../arch/Device.h"
#include "semantics/Semantics_fwd.h"

namespace ptx {
	namespace exec {
		class PtxExecutionContext : public gemu::AbstractExecutionContext {
		public:
			PtxExecutionContext (gemu::Device& device)
				:gemu::AbstractExecutionContext(device)
			{}
			~PtxExecutionContext(){

			}
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
		};
	}
}

#endif
