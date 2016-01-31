#ifndef PTXRETURNINSTRH
#define PTXRETURNINSTRH

#include "semantics/instructions/control/ControlInstruction.h"

namespace ptx {
	class Return : public ControlInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Return (bool isDivergent=true)
			:_isDivergent(isDivergent)
		{}
		bool isDivergent() const { return this->_isDivergent; }
        std::string toString() const override { return "<return>"; }
	private:
		bool _isDivergent;
	};
}

#endif
