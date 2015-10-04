#ifndef PTXINSTRVARDECLH
#define PTXINSTRVARDECLH

#include "semantics/Instruction.h"
#include "semantics/Variable.h"
#include "semantics/SymbolTable.h"

namespace ptx {
	class VariableDeclaration : public Instruction {
		PTX_DECLARE_DISPATCH
	public:
		VariableDeclaration(const ptx::Variable& toDecl)
			:_toDeclare(toDecl)
		{
		}
		ptx::Variable var() const {
			return this->_toDeclare;
		}
		std::string toString() const override {
			return "<declare> " + this->_toDeclare.name();
		}
        void declare(SymbolTable& symbols) const {
            int pos = -1;
            int pos2 = -1;
            const std::string name = this->_toDeclare.name();
            for (size_t i=0 ; i<name.length() ; ++i) {
                if (name[i] == '<') {
                    pos = i;
                } else if (name[i] == '>') {
                    pos2 = i;
                    break;
                }
            }
            if (pos > 0 && pos2 > pos) {
                const size_t count = atoi(name.substr(pos+1, pos2 - pos - 1).c_str());
                const std::string baseName = name.substr(0, pos);
                for (size_t i=1 ; i<count ; ++i) {
                    std::stringstream ss;
                    ss << baseName << i;
                    symbols.set(ptx::Variable(this->_toDeclare.space(),
                                              this->_toDeclare.type(),
                                              this->_toDeclare.size(),
                                              ss.str()),
                                param_storage_t());
                }
            } else {
                symbols.set(this->_toDeclare, param_storage_t());
            }
        }
	private:
		ptx::Variable _toDeclare;
	};
}

#endif
