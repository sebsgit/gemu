#include "semantics/Variable.h"
#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class VariableParser : public AbstractParser {
		public:
			bool parse(TokenList& tokens, ParserResult& result) const override {
				return true;
			}
		};
	}
}
