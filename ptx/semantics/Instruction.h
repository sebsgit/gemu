#ifndef PTXINSTRUCTIONH
#define PTXINSTRUCTIONH

#include <memory>

namespace ptx {
	class Instruction {
	public:
		virtual ~Instruction() {};
		virtual std::string toString() const{
			return std::string("[not implemented]");
		}
	protected:
	};
	typedef std::shared_ptr<ptx::Instruction> InstructionPtr;

	class ControlInstruction : public Instruction {

	};
	class MemoryInstruction : public Instruction {

	};
}

#endif
