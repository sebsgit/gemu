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
		bool empty() const {
			return this->_tokens.empty();
		}
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
		const token_t peek(size_t i=0) const {
			return i < this->_tokens.size() ? this->_tokens.at(i) : token_t();
		}
        bool poll(const token_t& token) {
            if (this->peek() == token){
                this->removeFirst();
                return true;
            }
            return false;
        }
        TokenList sublist(int start, int length) const;
        TokenList sublist(const token_t& startToken, const token_t& endToken) const;
        TokenList sublist(const token_t& endToken) const;
        TokenList bracketBody() const;
        void removeBracketBody();
        const token_t takeFirst() {
            const auto result = this->peek();
            this->removeFirst();
            return result;
        }
        token_t removeFirst(const size_t count = 1) {
            token_t result;
            if (count <= this->size()) {
                result = this->_tokens[0];
                this->_tokens.erase(this->_tokens.begin(), this->_tokens.begin() + count);
            }
            return result;
		}
		void removeUntil(const token_t& token) {
			const auto pos = std::find(this->_tokens.begin(), this->_tokens.end(), token);
			if (pos != this->_tokens.end())
				this->removeFirst(pos-this->_tokens.begin());
		}
		void removeUntilWith(const token_t& token) {
			const auto pos = std::find(this->_tokens.begin(), this->_tokens.end(), token);
			if (pos != this->_tokens.end())
				this->removeFirst(pos-this->_tokens.begin()+1);
		}
		void clear() {
			this->_tokens.clear();
		}
        void print() const {
            this->print(this->size());
        }
        void print(int maxTokens) const;
        std::string toString() const;
        std::vector<token_t>::iterator begin() { return this->_tokens.begin(); }
        std::vector<token_t>::iterator end() { return this->_tokens.end(); }
        std::vector<token_t>::const_iterator begin() const { return this->_tokens.cbegin(); }
        std::vector<token_t>::const_iterator end() const { return this->_tokens.cend(); }
		friend std::ostream& operator << (std::ostream& out, const TokenList& list);
	private:
		std::vector<token_t> _tokens;
	};

	extern std::ostream& operator << (std::ostream& out, const TokenList& list);

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
        void parse_line(std::string line, TokenList& result) const;
        bool is_bracket(const char c) const;
        std::string strip_multiline_comments(const std::string& source) const;
        void strip_singleline_comment(std::string& line) const;
	};
}

#endif
