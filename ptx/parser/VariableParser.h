#include "semantics/Variable.h"
#include "parser/AbstractParser.h"

namespace ptx {
	namespace parser {
		class VariableParser {
		public:
			bool parse(TokenList& tokens, std::vector<ptx::Variable>& result) const {
				return true;
			}
		};
	}
}
