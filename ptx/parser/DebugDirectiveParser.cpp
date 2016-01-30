#include "DebugDirectiveParser.h"
#include "semantics/globals.h"

namespace ptx {
namespace parser {
    bool DebugDirectiveParser::parseTokens(TokenList& tokens, ParserResult& result) const {
        if (tokens.poll(".loc")) {
            PTX_UNUSED(result);
            tokens.removeFirst(3);
            return true;
        }
        return false;
    }
}
}
