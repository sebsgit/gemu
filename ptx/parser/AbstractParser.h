#ifndef PTXABSTRACTPARSERH
#define PTXABSTRACTPARSERH

#include "parser/ParserUtils.h"
#include "parser/ParserResult.h"
#include "Tokenizer.h"

namespace ptx {
	namespace parser {
		class AbstractParser {
		protected:
            typedef TokenList::token_t token_t;
		public:
			virtual ~AbstractParser(){}
			bool parse(TokenList& tokens, ParserResult& result) const {
				ParserResult partialResult;
                const bool parsedOk = this->parseTokens(tokens, partialResult);
				if (parsedOk) {
					result.add(partialResult);
				}
				return parsedOk;
			}
		protected:
			virtual bool parseTokens(TokenList& tokens, ParserResult& result) const = 0;
			template <typename T>
			bool standardParse(const token_t& token, const int opCount, TokenList& tokens, ParserResult& result) const {
				if (tokens.poll(token)) {
					Type type;
					size_t size;
					if (Utils::parseTypeAndSize(tokens, &type, &size)) {
						MemoryInstructionOperands operands;
						if (Utils::parseOperands(tokens, opCount, &operands)) {
							MemoryInstruction instr;
							instr.setType(type);
							instr.setSize(size);
							instr.setOperands(operands);
							result.add(std::make_shared<T>(std::move(instr)));
							return true;
						}
					}
				}
				return false;
			}
		};

		template <typename Parser>
		class SplittingParser : public AbstractParser {
		private:
			const token_t _separator;
		public:
			SplittingParser(const token_t& token) : _separator(token) {

			}
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const override {
				if (tokens.empty())
					return true;
				Parser parser;
				bool parsedOk = false;
				while (!tokens.empty()) {
					if(!parser.parse(tokens, result)) {
						std::cout << "PARSER ERROR: " << tokens.peek() << " " << tokens.peek(1) << " " << tokens.peek(2) << '\n';
						return false;
					}
					if (tokens.empty()) {
						parsedOk = true;
						break;
                    } else if (tokens.poll(this->_separator)) {
						if (tokens.empty()) {
							parsedOk = true;
							break;
						}
					}
				}
				return parsedOk;
			}
		};
	}
}

#endif
