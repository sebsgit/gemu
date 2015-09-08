#ifndef SEMANTICGLOBALSH
#define SEMANTICGLOBALSH

namespace ptx {
	enum AllocSpace {
		reg,
		sreg,
		const_,
		global,
		local,
		param,
		shared,
		undefined
	};
	enum Type {
		Signed,
		Unsigned,
		Float,
		Bits,
		Predicate,
		Undefined
	};
}

#endif
