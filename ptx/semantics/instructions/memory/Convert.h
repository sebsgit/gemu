#ifndef PTXSEMANTICCONVERTINSTRH
#define PTXSEMANTICCONVERTINSTRH

#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Convert : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Convert(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<convert> " + MemoryInstruction::toString();
		}
        void resolve(SymbolTable& symbols) const override {
			symbols.set(this->_operands[0], symbols.get(this->_operands[1]));
        }
	};

	class ConvertValue : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		ConvertValue(MemoryInstruction&& other) : MemoryInstruction(std::move(other)) {}
		std::string toString() const override {
			return "<cvt> " + MemoryInstruction::toString();
		}
		void resolve(SymbolTable& symbols) const override {
			param_storage_t stored;
			auto source = symbols.get(this->_operands[1]);
			param_cast<float>(stored) = param_cast<unsigned>(source);
			symbols.set(this->_operands[0], stored);
		}
	};

}

#endif
