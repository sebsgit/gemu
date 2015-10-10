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
	if (name.size() > 2 && name[0]=='0' && name[1] == 'f'){
		const auto number = name.substr(2, name.size()-2);
		int i = strtol(number.c_str(), 0, 16);
		result.f = *(float*)(&i);
	} else {
		result.data = atoi(name.c_str());
	}
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

static bool test_predicate(const Instruction& i, const SymbolTable& symbols) {
	if (i.hasPredicate()) {
        const param_storage_t value = symbols.get(i.predicate());
        if (value.b == i.predicateNegated()) {
            return true;
        }
    }
	return false;
}

void PtxExecutionContext::exec(const MemoryInstruction& i) {
	if (!test_predicate(i, this->_symbols))
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
	if (!test_predicate(branch, this->_symbols) && this->_instr && this->_instr->hasLabel(branch.label()))
		this->_pc = this->_instr->instructionIndex(branch.label());
}
void PtxExecutionContext::exec(const VariableDeclaration& var) {
    var.declare(this->_symbols);
}

void PtxExecutionContext::exec(const Barrier& barrier) {
	//TODO implement barriers
}

using namespace gemu;
using namespace cuda;

PtxBlockDispatcher::PtxBlockDispatcher(gemu::Device& device, gemu::cuda::ThreadBlock& block)
:_device(device)
,_block(block)
{

}

static void alloc_constant(SymbolTable& symbols, const std::string& name, const int value) {
    param_storage_t tmp;
    tmp.i = value;
    symbols.set(ptx::Variable(AllocSpace::Constant,Type::Signed,sizeof(int),name),tmp);
}

static void set_grid_data(SymbolTable& symbols, const ThreadGrid& grid){
    alloc_constant(symbols, "%nctaid.x", grid.size().x);
    alloc_constant(symbols, "%nctaid.y", grid.size().y);
    alloc_constant(symbols, "%nctaid.z", grid.size().z);
}

static void set_block_data(SymbolTable& symbols, const ThreadBlock& block){
    alloc_constant(symbols, "%ntid.x", block.size().x);
    alloc_constant(symbols, "%ntid.y", block.size().y);
    alloc_constant(symbols, "%ntid.z", block.size().z);
}

static void set_thread_data(SymbolTable& symbols, int x, int y, int z) {
    alloc_constant(symbols, "%tid.x", x);
    alloc_constant(symbols, "%tid.y", y);
    alloc_constant(symbols, "%tid.z", z);
}

bool PtxBlockDispatcher::launch(ptx::Function& func, SymbolTable& symbols) {
	const dim3 size(this->_block.size());
	try {
        set_grid_data(symbols, this->_block.grid());
        set_block_data(symbols, this->_block);
		for (size_t x=0 ; x<size.x ; ++x) {
			for (size_t y=0 ; y<size.y ; ++y) {
				for (size_t z=0 ; z<size.z ; ++z) {
					Thread thread = this->_block.thread(x,y,z);
					SymbolTable symTab = symbols;
                    set_thread_data(symTab, x, y, z);
					PtxExecutionContext context(this->_device, thread, symTab);
					context.exec(func);
				}
			}
		}
	} catch (const std::exception& exc) {
        std::cout << exc.what() << '\n';
		return false;
	}
	return true;
}
