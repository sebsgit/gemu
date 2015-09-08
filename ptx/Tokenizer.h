#ifndef PTXTOKENIZERH
#define PTXTOKENIZERH

#include <vector>
#include <string>
#include <algorithm>
#include <sstream>

namespace ptx {

	class TokenList {
	public:
		typedef std::string token_t;
		size_t size() const {
			return this->_tokens.size();
		}
		TokenList& operator << (const token_t& token) {
			this->_tokens.push_back(token);
			return *this;
		}
		const token_t operator[](int i) const {
			return this->_tokens[i];
		}
		token_t& operator[](int i) {
			return this->_tokens[i];
		}
		const token_t peek(int i=0) const {
			return this->_tokens.at(i);
		}
		TokenList sublist(const token_t& startToken, const token_t& endToken) const {
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
		void removeFirst(const size_t count) {
			if (count <= this->size())
				this->_tokens.erase(this->_tokens.begin(), this->_tokens.begin() + count);
		}
		void removeUntil(const token_t& token) {
			const auto pos = std::find(this->_tokens.begin(), this->_tokens.end(), token);
			if (pos != this->_tokens.end())
				this->removeFirst(pos-this->_tokens.begin());
		}
		friend std::ostream& operator << (std::ostream& out, const TokenList& list);
	private:
		std::vector<token_t> _tokens;
	};

	std::ostream& operator << (std::ostream& out, const TokenList& list) {
		for (auto t : list._tokens)
			out << t << " ";
		out << "\n";
		return out;
	}

	class Tokenizer {
	public:
		TokenList tokenize(const std::string& source) const {
			TokenList result;
			std::string line;
			std::stringstream ss(this->strip_multiline_comments(source));
			while (std::getline(ss,line,'\n'))
				this->parse_line(line, result);
			return result;
		}
	private:
		#define PUSHT if (!token.empty()) { result << token; token.clear(); }
		void parse_line(std::string line, TokenList& result) const{
			this->strip_singleline_comment(line);
			// std::cout << "parsing line: " << line << "\n";
			std::string token;
			for (const char c : line) {
				if (std::isspace(c)){
					if (!token.empty()){
						result << token;
						token.clear();
						continue;
					}
				} else {
					if (is_bracket(c) || c==',' || c==':') {
						PUSHT
						token.push_back(c);
						PUSHT
					} else if (c == ';') {
						PUSHT
						result << ";";
					} else {
						token.push_back(c);
					}
				}
			}
			if (token.empty()==false)
				result << token;
		}
		#undef PUSHT
		bool is_bracket(const char c) const {
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
		std::string strip_multiline_comments(const std::string& source) const {
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
		void strip_singleline_comment(std::string& line) const {
			const size_t pos = line.find("//",0);
			if (pos != line.npos) {
				line = line.substr(0,pos);
			}
		}
	};
}

#endif
