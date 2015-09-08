#ifndef PTXDIRECTIVEPARSERH
#define PTXDIRECTIVEPARSERH

#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class DirectiveParser : public AbstractParser {
		public:
			~DirectiveParser(){}
			bool parse(TokenList& tokens, ParserResult& result) const {
				if (isDirective(tokens.peek())) {
					if (tokens.peek() == ".version") {
						result.module().setVersion(atof(tokens.peek(1).c_str()));
						tokens.removeFirst(2);
						return true;
					} else if (tokens.peek() == ".target") {
						result.module().setTargetArch(tokens.peek(1));
						tokens.removeFirst(2);
						return true;
					} else if (tokens.peek() == ".address_size") {
						result.module().setAddressSize(atoi(tokens.peek(1).c_str()));
						tokens.removeFirst(2);
						return true;
					}
				}
				return false;
			}
		private:
			bool isDirective(const token_t& token) const {
				return token.at(0) == '.';
			}
		};
	}
}

#endif
