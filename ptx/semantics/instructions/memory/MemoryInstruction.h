#ifndef PTXSEMANTICSMEMORYINSTRUCTIONBASEH
#define PTXSEMANTICSMEMORYINSTRUCTIONBASEH

#include "semantics/Instruction.h"

namespace ptx {
	class MemoryInstructionOperand {
	public:
		MemoryInstructionOperand(const std::string& name=std::string(), bool isAddressed=false, size_t offset=0)
		:_name(name)
		,_isAddressed(isAddressed)
		,_offset(offset)
		{}
		std::string symbol() const { return this->_name; }
		bool isAddressed() const { return this->_isAddressed; }
		size_t offset() const { return this->_offset; }
		std::string toString() const {
			return this->symbol() + (isAddressed() ? " [addres]" : "") + (offset() ? "[off]":"");
		}
	private:
		std::string _name;
		bool _isAddressed = false;
		size_t _offset = 0;
	};
	class MemoryInstructionOperands {
	public:
		size_t count() const { return this->_operands.size(); }
		const MemoryInstructionOperand& operator[](size_t i) const{
			return this->_operands[i];
		}
		void add(const MemoryInstructionOperand& op) { this->_operands.push_back(op); }
	private:
		std::vector<MemoryInstructionOperand> _operands;
	};

	class MemoryInstruction : public Instruction {
	public:
		MemoryInstructionOperands operands() const { return this->_operands; }
		CacheOperation cacheMode() const { return this->_cacheMode; }
		AllocSpace space() const { return this->_space; }
		Type type() const { return this->_type; }
		VectorType vectorized() const { return this->_vectorType; }
		size_t size() const { return this->_size; }
		bool isVolatile() const { return this->_isVolatile; }
		std::string toString() const override {
			std::stringstream ss;
			for (size_t i=0 ; i<this->_operands.count() ; ++i)
				ss << ':' << this->_operands[i].toString() << ':' << ", ";
			ss << "size: " << this->size();
			if (this->isVolatile())
				ss << " [volatile]";
			return ss.str();
		}
		void addOperand(const MemoryInstructionOperand& op) { this->_operands.add(op); }
		void setOperands(const MemoryInstructionOperands& ops) { this->_operands = ops; }
		void setCacheMode(CacheOperation op) { this->_cacheMode = op; }
		void setAllocSpace(AllocSpace space) { this->_space = space; }
		void setType(Type type) { this->_type = type; }
		void setVectorized(VectorType vec) { this->_vectorType = vec; }
		void setSize(const size_t s) { this->_size = s; }
		void setVolatile(bool v=true) { this->_isVolatile = v; }
	protected:
		MemoryInstructionOperands _operands;
		CacheOperation _cacheMode = CacheOperation::CacheAllLevels;
		AllocSpace _space = AllocSpace::Undefined;
		Type _type = Type::Unknown;
		VectorType _vectorType = VectorType::VecNone;
		size_t _size=0;
		bool _isVolatile = false;
	};
}

#endif
