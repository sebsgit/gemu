#include "runtime/PtxExecutionContext.h"
#include "semantics/Semantics.h"
#include <sstream>
#include <string>
#include <cstring>
#include <iostream>

using namespace ptx;
using namespace exec;

void SymbolTable::set(const std::string& name, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end()) {
		it->data = storage;
	}
}
bool SymbolTable::has(const std::string& name) const {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	return it != _data.end();
}
param_storage_t SymbolTable::get(const std::string& name) const {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end())
		return it->data;
	return param_storage_t();
}
void SymbolTable::set(const ptx::Variable& var, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == var.name();});
	if (it != _data.end()) {
		it->data = storage;
	} else {
		_data.push_back(entry_t(var, storage));
	}
}
bool SymbolTable::has(const ptx::Variable& var) const {
	return this->has(var.name());
}
param_storage_t SymbolTable::get(const ptx::Variable& var) const {
	return this->get(var.name());
}

void PtxExecutionContext::exec(const InstructionList& list) {
	for (size_t i=0 ; i<list.count() ; ++i)
		list.fetch(i)->dispatch(*this);
}

void PtxExecutionContext::exec(const Instruction& i) {
	std::cout << "!!!EXEC DEFAULT: " << i.toString() << "\n";
	exit(0);
}

static void load_impl(const MemoryInstructionOperand& to,
					  const MemoryInstructionOperand& from,
					  SymbolTable& symbols,
				  	  const size_t size)
{
	param_storage_t stored;
	param_storage_t source = symbols.get(from.symbol());
	memcpy(&stored, &source, size / 8);
	symbols.set(to.symbol(), stored);
}

void PtxExecutionContext::exec(const Load& load) {
	// std::cout << "exec load: " << load.toString() << "\n";
	load_impl(load.operands()[0], load.operands()[1], this->_symbols, load.size());
}

static void store_impl(const Store& store, SymbolTable& symbols) {
	const param_storage_t dest = symbols.get(store.operands()[0].symbol());
	const param_storage_t source = symbols.get(store.operands()[1].symbol());;
	*(reinterpret_cast<unsigned int *>(dest.data)) = source.data;
}

void PtxExecutionContext::exec(const Store& store) {
	// std::cout << "exec store: " << store.toString() << "\n";
	store_impl(store, this->_symbols);
}

static void move_impl(const Move& move, SymbolTable& symbols) {
	param_storage_t dest = symbols.get(move.operands()[0].symbol());
	param_storage_t source;
	const std::string srcName = move.operands()[1].symbol();
	if (symbols.has(srcName)) {
		source = symbols.get(srcName);
	} else {
		//TODO: LITERAL
		source.data = atoi(srcName.c_str());
	}
	memcpy(&dest, &source, move.size() / 8);
	symbols.set(move.operands()[0].symbol(), dest);
}

void PtxExecutionContext::exec(const Move& move) {
	move_impl(move, this->_symbols);
}
void PtxExecutionContext::exec(const Return& r) {
	// std::cout << "exec return: " << r.toString() << "\n";
}
void PtxExecutionContext::exec(const Convert& conv) {
	// std::cout << "exec conv: " << conv.toString() << "\n";
	load_impl(conv.operands()[0], conv.operands()[1], this->_symbols, conv.size());
}
void PtxExecutionContext::exec(const FunctionDeclaration& fdecl) {
	// std::cout << "exec func decl: " << fdecl.toString() << "\n";
}
void PtxExecutionContext::exec(const ModuleDirective& d) {
	// std::cout << "exec directive: " << d.toString() << "\n";
}

static void declare_var(const ptx::Variable& var, SymbolTable& symbols) {
	int pos = -1;
	int pos2 = -1;
	const std::string name = var.name();
	for (size_t i=0 ; i<name.length() ; ++i) {
		if (name[i] == '<') {
			pos = i;
		} else if (name[i] == '>') {
			pos2 = i;
			break;
		}
	}
	if (pos > 0 && pos2 > pos) {
		const size_t count = atoi(name.substr(pos+1, pos2 - pos - 1).c_str());
		const std::string baseName = name.substr(0, pos);
		for (size_t i=0 ; i<count ; ++i) {
			std::stringstream ss;
			ss << baseName << i;
			symbols.set(ptx::Variable(var.space(), var.type(), var.size(), ss.str()), param_storage_t());
		}
	} else {
		symbols.set(var, param_storage_t());
	}
}
void PtxExecutionContext::exec(const VariableDeclaration& var) {
	// std::cout << "exec var decl: " << var.toString() << "\n";
	declare_var(var.var(), this->_symbols);
}


using namespace gemu;
using namespace cuda;

PtxBlockDispatcher::PtxBlockDispatcher(gemu::Device& device, gemu::cuda::ThreadBlock& block)
:_device(device)
,_block(block)
{

}

bool PtxBlockDispatcher::launch(ptx::Function& func, SymbolTable& symbols) {
	const dim3 size(this->_block.size());
	try {
		for (size_t x=0 ; x<size.x ; ++x) {
			for (size_t y=0 ; y<size.y ; ++y) {
				for (size_t z=0 ; z<size.z ; ++z) {
					Thread thread = this->_block.thread(x,y,z);
					SymbolTable symTab = symbols;
					PtxExecutionContext context(this->_device, thread, symTab);
					context.exec(func);
				}
			}
		}
	} catch (const std::exception& exc) {
		return false;
	}
	return true;
}
