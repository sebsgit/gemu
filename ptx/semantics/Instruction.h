#ifndef PTXINSTRUCTIONH
#define PTXINSTRUCTIONH

#include <memory>

namespace ptx {
	class Instruction {
	public:
		virtual ~Instruction() {};
		virtual std::string toString() const{
			return std::string("[not implemented]");
		}
	protected:
	};
	typedef std::shared_ptr<ptx::Instruction> InstructionPtr;

	class ModuleDirective : public Instruction {
	public:
		enum Type {
			Undefined,
			Version,
			Target,
			AddressSize
		};
		ModuleDirective(Type t, const std::string& data)
			:_type(t)
			,_data(data)
		{}
		float version() const{
			if (this->_type == Version)
				return atof(this->_data.c_str());
			return -1.0f;
		}
		int addressSize() const {
			if (this->_type == AddressSize)
				return atoi(this->_data.c_str());
			return -1;
		}
		std::string target() const {
			if (this->_type == Target)
				return this->_data;
			return std::string();
		}
		std::string toString() const override {
			switch (this->_type){
				case Version: return ".version " + this->_data;
				case Target: return ".target " + this->_data;
				case AddressSize: return ".address_size " + this->_data;
				default: break;
			}
			return Instruction::toString();
		}
	private:
		Type _type;
		std::string _data;
	};
	class VariableDeclaration : public Instruction {

	};
	class FunctionDeclaration : public Instruction {

	};
	class ControlInstruction : public Instruction {

	};
	class MemoryInstruction : public Instruction {

	};
}

#endif
