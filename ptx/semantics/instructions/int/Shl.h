#ifndef PTXSEMANTINTSHLLH
#define PTXSEMANTINTSHLLH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Shl : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Shl(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<shl> " + MemoryInstruction::toString();
		}
//        void resolve(SymbolTable& symbols) const override {

//        }
	};

    class Shr : public MemoryInstruction {
        PTX_DECLARE_DISPATCH
    public:
        Shr(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
        std::string toString() const override {
            return "<shr> " + MemoryInstruction::toString();
        }
		void resolve(SymbolTable& symbols) const override {
			param_storage_t dest;
			const auto src = symbols.get(this->_operands[1]);
			const int shiftby = param_cast<unsigned>(symbols.get(this->_operands[2]));
			switch(this->type()) {
			case Type::Signed:
				if (this->size() < 8)
					param_cast<int>(dest) = param_cast<int>(src) >> shiftby;
				else
					param_cast<long long>(dest) = param_cast<long long>(src) >> shiftby;
				break;
			case Type::Unsigned:
				if (this->size() < 8)
					param_cast<unsigned int>(dest) = param_cast<unsigned int>(src) >> shiftby;
				else
					param_cast<unsigned long long>(dest) = param_cast<unsigned long long>(src) >> shiftby;
				break;
			default:
				//TODO error
				break;
			}

			symbols.set(this->_operands[0], dest);
		}
    };
}

#endif
