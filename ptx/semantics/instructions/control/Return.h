#ifndef PTXRETURNINSTRH
#define PTXRETURNINSTRH

#include "semantics/instructions/control/ControlInstruction.h"

namespace ptx {
	class Return : public ControlInstruction {
	public:
		Return (bool isDivergent=true)
			:_isDivergent(isDivergent)
		{}
		bool isDivergent() const { return this->_isDivergent; }
	private:
		bool _isDivergent;
	};
}

#endif
