#ifndef PTXSYMBLTABLEH
#define PTXSYMBLTABLEH

#include "semantics/Variable.h"
#include <vector>

namespace ptx {
	union param_storage_t {
		unsigned long long data = 0;
        bool b;
        int i;
        long l;
        long long ll;
        float f;
        double d;
		unsigned u;
        unsigned long ul;
        unsigned long long ull;
	};

	class SymbolTable {
		struct entry_t {
			ptx::Variable var;
			param_storage_t data;
			entry_t(const ptx::Variable& v=ptx::Variable(), const param_storage_t& d=param_storage_t())
			:var(v)
			,data(d)
			{}
		};
	public:
		void set(const ptx::Variable& var, const param_storage_t& storage);
		void set(const std::string& name, const param_storage_t& storage);
		bool has(const ptx::Variable& var) const;
		bool has(const std::string& name) const;
		param_storage_t get(const ptx::Variable& var) const;
		param_storage_t get(const std::string& name) const;

		void print() const{
			for (const auto& x: _data)
				std::cout << x.var.name() << ": " << x.data.data << "\n";
		}

	private:
		std::vector<entry_t> _data;
	};
}

#endif
