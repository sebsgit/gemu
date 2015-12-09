#include "InstructionParser.h"
#include "parser/VariableParser.h"
#include "parser/StoreParser.h"
#include "parser/LoadParser.h"
#include "parser/MoveParser.h"
#include "parser/ConvertParser.h"
#include "parser/ReturnExitParser.h"
#include "parser/BranchParser.h"
#include "parser/SetpParser.h"
#include "parser/SelpParser.h"
#include "parser/MulParser.h"
#include "parser/AddParser.h"
#include "parser/AndParser.h"
#include "parser/OrParser.h"
#include "parser/MadParser.h"
#include "parser/DivParser.h"
#include "parser/SubParser.h"
#include "parser/ShlParser.h"
#include "parser/BarrierParser.h"
#include "parser/AtomicParser.h"
#include "parser/CallParser.h"
#include "parser/RemParser.h"

namespace ptx {
    namespace parser {
        InstructionParser::InstructionParser(){
            parsers.push_back(std::make_shared<VariableParser>());
            parsers.push_back(std::make_shared<StoreParser>());
            parsers.push_back(std::make_shared<LoadParser>());
            parsers.push_back(std::make_shared<MoveParser>());
            parsers.push_back(std::make_shared<ConvertParser>());
            parsers.push_back(std::make_shared<ReturnExitParser>());
            parsers.push_back(std::make_shared<BranchParser>());
            parsers.push_back(std::make_shared<SetpParser>());
            parsers.push_back(std::make_shared<SelpParser>());
            parsers.push_back(std::make_shared<MulParser>());
            parsers.push_back(std::make_shared<DivParser>());
            parsers.push_back(std::make_shared<AddParser>());
            parsers.push_back(std::make_shared<AndParser>());
            parsers.push_back(std::make_shared<OrParser>());
            parsers.push_back(std::make_shared<XorParser>());
            parsers.push_back(std::make_shared<MadParser>());
            parsers.push_back(std::make_shared<RemParser>());
            parsers.push_back(std::make_shared<SubParser>());
            parsers.push_back(std::make_shared<ShlParser>());
            parsers.push_back(std::make_shared<BarrierParser>());
            parsers.push_back(std::make_shared<AtomicParser>());
            parsers.push_back(std::make_shared<InstructionBracketParser>());
            parsers.push_back(std::make_shared<CallParser>());
        }
        bool InstructionParser::parseTokens(TokenList& tokens, ParserResult& result) const {
            bool parsedOk = false;
            std::string label;
            std::string predicate;
            bool predicateNegated=false;
            Utils::parseLabel(tokens, &label);
            Utils::parsePredicate(tokens, &predicate, &predicateNegated);
            for (const auto& p : this->parsers) {
                if (p->parse(tokens, result)) {
                    if (label.empty()==false)
                        result.labelLast(label);
                    if (predicate.empty()==false)
                        result.predicateLast(predicate, predicateNegated);
                    parsedOk = true;
                    break;
                }
            }
            return parsedOk;
        }
        InstructionBlockParser::InstructionBlockParser()
            :SplittingParser<InstructionParser>(token_t(";"))
        {}
    }
}
