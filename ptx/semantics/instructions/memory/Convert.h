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
            param_storage_t stored;
			param_storage_t source = symbols.get(this->_operands[1]);
            memcpy(&stored, &source, this->size());
			symbols.set(this->_operands[0], stored);
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
			param_storage_t source = symbols.get(this->_operands[1]);
			stored.f = source.u;
			symbols.set(this->_operands[0], stored);
		}
	};

}

#endif
