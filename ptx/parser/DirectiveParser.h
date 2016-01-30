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
                    TokenList::token_t value;
					ptx::ModuleDirective::Type type = ptx::ModuleDirective::Undefined;
                    if (tokens.poll(".version")) {
						type = ptx::ModuleDirective::Version;
                        value = tokens.removeFirst();
                    } else if (tokens.poll(".target")) {
						type = ptx::ModuleDirective::Target;
                        value = tokens.removeFirst();
                    } else if (tokens.poll(".address_size")) {
						type = ptx::ModuleDirective::AddressSize;
                        value = tokens.removeFirst();
                    } else if (tokens.poll(".file")) {
                        tokens.removeFirst(); // file index
                        if (tokens.peek().find('\"') != std::string::npos) {
                            int filenameEndPos = -1;
                            for (size_t i=1 ; i<tokens.size() ; ++i) {
                                if (tokens.peek(i).find('\"') != std::string::npos){
                                    filenameEndPos = i;
                                    break;
                                }
                            }
                            if (filenameEndPos > -1) {
                                tokens.removeUntilWith(tokens.peek(filenameEndPos));
                                if (tokens.peek() == "," && tokens.peek(2) == ",")
                                    tokens.removeFirst(4);
                                type = ptx::ModuleDirective::FileName;
                            }
                        }
                    }
					if (type != ptx::ModuleDirective::Undefined) {
                        result.add(std::make_shared<ptx::ModuleDirective>(type, value));
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
