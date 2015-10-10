#ifndef PTXBARIERINSTRH
#define PTXBARIERINSTRH

#include "semantics/instructions/control/ControlInstruction.h"
#include "semantics/instructions/memory/MemoryInstruction.h"

namespace ptx {
	class Barrier : public MemoryInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Barrier (int id, BarrierType type, MemoryInstruction&& other)
			: MemoryInstruction(std::move(other))
			, _type(type)
			, _id(id)
		{}
        std::string toString() const override {
            return "<barrier> ";
        }
		void setId(const int id){this->_id = id;}
		int id() const {return this->_id;}
		void setType(BarrierType type) {this->_type = type;}
		BarrierType type() const { return this->_type; }
	private:
		BarrierType _type;
		int _id;
	};
}

#endif
