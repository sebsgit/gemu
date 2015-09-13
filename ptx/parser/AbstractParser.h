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
				TokenList copy = tokens;
				ParserResult partialResult;
				const bool parsedOk = this->parseTokens(copy, partialResult);
				if (parsedOk) {
					tokens = copy;
					result.add(partialResult);
				}
				return parsedOk;
			}
		protected:
			virtual bool parseTokens(TokenList& tokens, ParserResult& result) const = 0;
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
					parser.parse(tokens, result);
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
				return parsedOk;
			}
		};
	}
}

#endif
