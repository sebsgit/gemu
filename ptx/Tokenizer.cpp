#include "Tokenizer.h"

#include <iostream>
#include <algorithm>

namespace ptx {

    TokenList TokenList::sublist(const token_t& startToken, const token_t& endToken) const {
        TokenList result;
        auto start = std::find(this->_tokens.begin(), this->_tokens.end(), startToken);
        if (start != this->_tokens.end()) {
            ++start;
            const auto end = std::find(this->_tokens.begin(), this->_tokens.end(), endToken);
            if (start != this->_tokens.end() && end != this->_tokens.end()) {
                std::copy(start, end, std::back_inserter(result._tokens));
            }
        }
        return result;
    }
    TokenList TokenList::sublist(const token_t& endToken) const {
        return this->sublist(this->_tokens.at(0), endToken);
    }
    TokenList TokenList::bracketBody() const {
        TokenList result;
        int i=1;
        int bracketBalance = this->peek() == "{";
        while (bracketBalance != 0) {
            if (this->_tokens[i] == "{")
                ++bracketBalance;
            else if (this->_tokens[i] == "}")
                --bracketBalance;
            if (bracketBalance == 0)
                break;
            result._tokens.push_back(this->_tokens[i]);
            ++i;
        }
        return result;
    }
    void TokenList::removeBracketBody() {
        int bracketBalance = this->peek()=="{";
        while (bracketBalance){
            this->removeFirst();
            if (this->peek() == "{")
                ++bracketBalance;
            else if (this->peek() == "}")
                --bracketBalance;
        }
        if(this->peek() == "}")
            this->removeFirst();
    }

    void TokenList::print() const{
        for(const auto & t : _tokens)
            std::cout << '"' << t << "\" ";
        std::cout << '\n';
    }


    #define PUSHT if (!token.empty()) { result << token; token.clear(); }
    void Tokenizer::parse_line(std::string line, TokenList& result) const{
        this->strip_singleline_comment(line);
        // std::cout << "parsing line: " << line << "\n";
        std::string token;
        for (size_t i=0 ; i<line.size() ; ++i) {
            const char previous = i > 0 ? line[i-1] : 0;
            const char c = line[i];
            const char next = i < line.size() - 1 ? line[i+1] : 0;
            if (std::isspace(c)){
                if (!token.empty()){
                    result << token;
                    token.clear();
                    continue;
                }
            } else {
                if (is_bracket(c) || c==',' || c==':' || c=='+') {
                    PUSHT
                    token.push_back(c);
                    PUSHT
                } else if (c == ';') {
                    PUSHT
                    result << ";";
                } else if (c == '.') {
                    if (!std::isdigit(next) || !std::isdigit(previous)){
                        PUSHT
                    }
                    token.push_back(c);
                } else {
                    token.push_back(c);
                }
            }
        }
        if (token.empty()==false)
            result << token;
    }
    #undef PUSHT

    bool Tokenizer::is_bracket(const char c) const {
        switch (c) {
            case '(':
            case ')':
            case '{':
            case '}':
            case '[':
            case ']':
            return true;
            default:
            return false;
        }
    }
    std::string Tokenizer::strip_multiline_comments(const std::string& source) const {
        std::string result = source;
        while (1) {
            const size_t posStart = result.find("/*", 0);
            size_t posEnd = result.npos;
            if (posStart != result.npos){
                posEnd = result.find("*/",posStart);
                if (posEnd != result.npos) {
                    result = result.substr(0, posStart) + result.substr(posEnd+2, result.size()-posEnd);
                }
            }
            if (posStart==result.npos || posEnd)
                break;
        }
        return result;
    }
    void Tokenizer::strip_singleline_comment(std::string& line) const {
        const size_t pos = line.find("//",0);
        if (pos != line.npos) {
            line = line.substr(0,pos);
        }
    }

}
