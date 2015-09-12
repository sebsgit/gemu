#include "semantics/Variable.h"
#include "parser/AbstractParser.h"
#include "semantics/instructions/VariableDeclaration.h"

namespace ptx {
	namespace parser {
		class VariableParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				AllocSpace space = parseAllocSpace(tokens.peek());
				if (space != AllocSpace::undefined) {
					ptx::Type type;
					size_t size = 0;
					parseTypeAndSize(tokens.peek(1), &type, &size);
					if  (size > 0) {
						std::string name = tokens.peek(2);
						tokens.removeFirst(3);
						result.add(std::make_shared<ptx::VariableDeclaration>(ptx::Variable(space, type, size, name)));
						if (tokens.peek() == ";")
							tokens.removeFirst();
						return true;
					}
				}
				return false;
			}
		private:
			static AllocSpace parseAllocSpace(const std::string& str) {
				if (str == ".reg")
					return AllocSpace::reg;
				else if (str == ".sreg")
					return AllocSpace::sreg;
				else if (str == ".const")
					return AllocSpace::const_;
				else if (str == ".global")
					return AllocSpace::global;
				else if (str == ".local")
					return AllocSpace::local;
				else if (str == ".param")
					return AllocSpace::param;
				else if (str == ".shared")
					return AllocSpace::shared;
				return AllocSpace::undefined;
			}
			static void parseTypeAndSize(const std::string& str, Type * type, size_t * size) {
				*size = 0;
				*type = Type::Undefined;
				if (str.size() > 2) {
					if (str == ".pred") {
						*type = Type::Predicate;
						*size = 1;
					} else {
						const char c = str[1];
						if (c == 'u')
							*type = Type::Unsigned;
						else if (c == 's')
							*type = Type::Signed;
						else if (c == 'b')
							*type = Type::Bits;
						else if (c == 'f')
							*type = Type::Float;
						const std::string sizeStr = str.substr(2,str.size()-2);
						*size = atoi(sizeStr.c_str());
						if (*size != 8 && *size!=16 && *size!=32 && *size!=64)
							*size = 0;
					}
				}
			}
		};

		class VariableListParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				if (tokens.empty())
					return true;
				const TokenList toRevert(tokens);
				ParserResult partialResult;
				VariableParser vars;
				bool parsedOk = false;
				while (!tokens.empty()) {
					vars.parse(tokens, partialResult);
					if (tokens.empty()) {
						parsedOk = true;
						break;
					} else if (tokens.peek() == ",") {
						tokens.removeFirst();
					}
				}
				if (parsedOk==false) {
					tokens = toRevert;
				} else {
					result.add(partialResult);
				}
				return parsedOk;
			}
		};
	}
}
