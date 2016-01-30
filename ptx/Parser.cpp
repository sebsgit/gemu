#include "Parser.h"
#include "semantics/Semantics.h"
#include "parser/DirectiveParser.h"
#include "parser/FunctionParser.h"

#include <iostream>
#include <cassert>
#include <algorithm>
#include <sstream>

namespace ptx {

ParserResult Parser::parseModule(const std::string& source){
    ParserResult result;
    auto tokens = Tokenizer().tokenize(source);
    if(!this->parse(tokens, result)) {
        std::cout << "\nparser error: " << tokens.sublist(0, 10).toString() << "\n\n";
        result = ParserResult();
    }
    return result;
}

bool Parser::parseTokens(TokenList& tokens, ParserResult& result) const {
    parser::DirectiveParser directive;
    while (directive.parse(tokens, result));
    parser::VariableParser globalVariables;
    while (globalVariables.parse(tokens, result));
    parser::FunctionParser functions;
    while (functions.parse(tokens, result));
    return tokens.empty();
}

}
