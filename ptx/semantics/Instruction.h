#ifndef PTXINSTRUCTIONH
#define PTXINSTRUCTIONH

#include <memory>

namespace ptx {
	class Instruction {
	public:
		virtual ~Instruction() {};
	protected:
	};
	typedef std::shared_ptr<ptx::Instruction> InstructionPtr;

	class ModuleDirective : public Instruction {

	};
	class VariableDeclaration : public Instruction {

	};
	class FunctionDeclaration : public Instruction {

	};
	class ControlInstruction : public Instruction {

	};
	class MemoryInstruction : public Instruction {

	};
}

#endif
