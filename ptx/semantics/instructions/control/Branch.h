#ifndef PTXBRANCHINSTRH
#define PTXBRANCHINSTRH

#include "semantics/instructions/control/ControlInstruction.h"

namespace ptx {
	class Branch : public ControlInstruction {
		PTX_DECLARE_DISPATCH
	public:
		Branch (const std::string& label=std::string(), bool isDivergent=false)
			:_label(label)
			,_isDivergent(isDivergent)
		{}
		bool isDivergent() const { return this->_isDivergent; }
		std::string label() const { return this->_label; }
	private:
		std::string _label;
		bool _isDivergent;
	};
}

#endif
