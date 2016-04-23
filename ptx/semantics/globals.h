#ifndef SEMANTICGLOBALSH
#define SEMANTICGLOBALSH

#define PTX_UNUSED(x) (void)(x)

namespace ptx {
	enum class AllocSpace {
		Register,
		SpecialRegister,
		Constant,
		Global,
		Local,
		Parameter,
		Shared,
		Undefined
	};
	enum class Type {
		Signed,
		Unsigned,
		Float,
		Bits,
		Predicate,
		Unknown
	};
	enum class VectorType {
		VecNone,
		Vec2,
		Vec4
	};
	enum class CacheOperation {
		CacheAllLevels,
		CacheGlobal,
		CacheStreaming,
		CacheLastUse,
		CacheVolatile,
		CacheUndefined,
		CacheWriteBack,
		CacheWriteThrough
	};
	enum class CompareOperation {
		NotValidCompareOperation,
		Equal,
		NotEqual,
		Lower,
		Greater,
		LowerEqual,
		GreaterEqual
	};
	enum class BooleanOperation {
		NotValidBooleanOperation,
		And,
		Or,
		Xor
	};
	enum class BarrierType {
		BarSync,
		BarArrive,
		BarReduction
	};
}

#endif
