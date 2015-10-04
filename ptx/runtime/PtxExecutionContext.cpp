#include "runtime/PtxExecutionContext.h"
#include "semantics/Semantics.h"

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
	//TODO literal
	param_storage_t result;
	result.data = atoi(name.c_str());
	return result;
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
	this->_instr = &list;
	while (this->_pc < list.count())
		list.fetch(this->_pc++)->dispatch(*this);
	this->_instr = nullptr;
}

void PtxExecutionContext::exec(const Instruction& i) {
	std::cout << "!!!EXEC DEFAULT: " << i.toString() << "\n";
	exit(0);
}

void PtxExecutionContext::exec(const MemoryInstruction& i) {
	i.resolve(this->_symbols);
}

void PtxExecutionContext::exec(const Return& r) {
	// std::cout << "exec return: " << r.toString() << "\n";
	if (this->_instr)
		this->_pc = this->_instr->count();
}
void PtxExecutionContext::exec(const FunctionDeclaration& fdecl) {
	// std::cout << "exec func decl: " << fdecl.toString() << "\n";
}
void PtxExecutionContext::exec(const ModuleDirective& d) {
	// std::cout << "exec directive: " << d.toString() << "\n";
}

void PtxExecutionContext::exec(const Branch& branch) {
	// std::cout << "jump to " << branch.label() << '\n';
	if (this->_instr && this->_instr->hasLabel(branch.label()))
		this->_pc = this->_instr->instructionIndex(branch.label());
}
void PtxExecutionContext::exec(const VariableDeclaration& var) {
    var.declare(this->_symbols);
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
