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
		void resolve(SymbolTable& symbols) const override {
			param_storage_t dest;
			const auto src = symbols.get(this->_operands[1]);
			const int shiftby = param_cast<unsigned>(symbols.get(this->_operands[2]));
			switch(this->type()) {
			case Type::Bits:
				if (this->size() < 8)
					this->do_shift<unsigned int>(dest, src, shiftby);
				else
					this->do_shift<unsigned long long>(dest, src, shiftby);
				break;
			default:
				//TODO error
				break;
			}

			symbols.set(this->_operands[0], dest);
		}
	private:
		template <typename T>
		void do_shift(param_storage_t& dest, const param_storage_t& src, int shiftby) const noexcept {
			if (shiftby >= sizeof(T) * 8)
				param_cast<T>(dest) = param_cast<T>(src) > 0 ? (T)0 : (T)~0;
			else
				param_cast<T>(dest) = param_cast<T>(src) << shiftby;
		}
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
					this->do_shift<int>(dest, src, shiftby);
				else
					this->do_shift<long long>(dest, src, shiftby);
				break;
			case Type::Unsigned:
				if (this->size() < 8)
					this->do_shift<unsigned int>(dest, src, shiftby);
				else
					this->do_shift<unsigned long long>(dest, src, shiftby);
				break;
			default:
				//TODO error
				break;
			}

			symbols.set(this->_operands[0], dest);
		}
	private:
		template <typename T>
		void do_shift(param_storage_t& dest, const param_storage_t& src, int shiftby) const noexcept {
			if (shiftby >= sizeof(T) * 8)
				param_cast<T>(dest) = param_cast<T>(src) > 0 ? (T)0 : (T)~0;
			else
				param_cast<T>(dest) = param_cast<T>(src) >> shiftby;
		}
    };
}

#endif
