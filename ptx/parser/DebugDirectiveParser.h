#ifndef PTXDEBUGDIRECTIVEPARSERH
#define PTXDEBUGDIRECTIVEPARSERH

#include "parser/AbstractParser.h"

namespace ptx {
    namespace parser {
    class DebugDirectiveParser: public AbstractParser {
    protected:
        bool parseTokens(TokenList& tokens, ParserResult& result) const override;
    };
    }
}

#endif
