#ifndef PTXDIRECTIVEPARSERH
#define PTXDIRECTIVEPARSERH

#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class DirectiveParser : public AbstractParser {
		public:
			~DirectiveParser(){}
		protected:
			bool parseTokens(TokenList& tokens, ParserResult& result) const {
				if (isDirective(tokens.peek())) {
					ptx::ModuleDirective::Type type = ptx::ModuleDirective::Undefined;
					if (tokens.peek() == ".version") {
						type = ptx::ModuleDirective::Version;
					} else if (tokens.peek() == ".target") {
						type = ptx::ModuleDirective::Target;
					} else if (tokens.peek() == ".address_size") {
						type = ptx::ModuleDirective::AddressSize;
					}
					if (type != ptx::ModuleDirective::Undefined) {
						result.add(std::make_shared<ptx::ModuleDirective>(type, tokens.peek(1)));
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
