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
			virtual bool parse(TokenList& tokens, ParserResult& result) const = 0;
		};

		template <typename Parser>
		class SplittingParser : public AbstractParser {
		private:
			const token_t _separator;
		public:
			SplittingParser(const token_t& token) : _separator(token) {

			}
			bool parse(TokenList& tokens, ParserResult& result) const override {
				if (tokens.empty())
					return true;
				const TokenList toRevert(tokens);
				ParserResult partialResult;
				Parser parser;
				bool parsedOk = false;
				while (!tokens.empty()) {
					parser.parse(tokens, partialResult);
					if (tokens.empty()) {
						parsedOk = true;
						break;
					} else if (tokens.peek() == this->_separator) {
						tokens.removeFirst();
						if (tokens.empty()) {
							parsedOk = true;
							break;
						}
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

#endif
