#ifndef PTXINSTRUCTIONH
#define PTXINSTRUCTIONH

#include <memory>

namespace ptx {
	class Instruction {
	public:
		virtual ~Instruction() {};
	};

	typedef std::shared_ptr<ptx::Instruction> InstructionPtr;
}

#endif
