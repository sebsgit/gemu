#ifndef SEMANTICGLOBALSH
#define SEMANTICGLOBALSH

namespace ptx {
	enum AllocSpace {
		Register,
		SpecialRegister,
		Constant,
		Global,
		Local,
		Parameter,
		Shared,
		Undefined
	};
	enum Type {
		Signed,
		Unsigned,
		Float,
		Bits,
		Predicate,
		Unknown
	};
	enum VectorType {
		VecNone,
		Vec2,
		Vec4
	};
	enum CacheOperation {
		CacheAllLevels,
		CacheGlobal,
		CacheStreaming,
		CacheLastUse,
		CacheVolatile,
		CacheUndefined,
		CacheWriteBack,
		CacheWriteThrough
	};
	enum CompareOperation {
		NotValidCompareOperation,
		Equal,
		NotEqual,
		Lower,
		Greater,
		LowerEqual,
		GreaterEqual
	};
	enum BooleanOperation {
		NotValidBooleanOperation,
		And,
		Or,
		Xor
	};
	enum BarrierType {
		BarSync,
		BarArrive,
		BarReduction
	};
}

#endif
