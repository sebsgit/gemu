#include "runtime/PtxExecutionContext.h"
#include "semantics/Semantics.h"
#include <thread>
#include <mutex>

using namespace ptx;
using namespace exec;

void SymbolStorage::set(const std::string& name, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end()) {
		it->data = storage;
	}
}
bool SymbolStorage::has(const std::string& name) const {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	return it != _data.end();
}

ptx::Variable SymbolStorage::variable(const std::string& name) const {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end())
		return it->var;
	return ptx::Variable();
}

unsigned long long SymbolStorage::address(const std::string& name) const{
	for (size_t i=0 ; i<this->_data.size() ; ++i){
		if (this->_data[i].var.name() == name) {
			return reinterpret_cast<unsigned long long>(&this->_data[i].data);
		}
	}
	return 0;
}

param_storage_t SymbolStorage::get(const std::string& name) const {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end())
		return it->data;
	//TODO literal
	param_storage_t result;
	if (name.size() > 2 && name[0]=='0' && name[1] == 'f'){
		const auto number = name.substr(2, name.size()-2);
		const int i = strtol(number.c_str(), 0, 16);
		result.f = *reinterpret_cast<const float*>(&i);
	} else {
		result.data = atoi(name.c_str());
	}
	return result;
}
void SymbolStorage::set(const ptx::Variable& var, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == var.name();});
	if (it != _data.end()) {
		it->data = storage;
	} else {
		_data.push_back(entry_t(var, storage));
	}
}
bool SymbolStorage::has(const ptx::Variable& var) const {
	return this->has(var.name());
}
param_storage_t SymbolStorage::get(const ptx::Variable& var) const {
	return this->get(var.name());
}

void SymbolTable::setSharedSection(ProtectedStoragePtr sharedData){
	this->_sharedData = sharedData;
}

void PtxExecutionContext::setProgramCounter(size_t pc) {
	this->_pc = pc;
}

size_t PtxExecutionContext::programCounter() const{
	return this->_pc;
}

void PtxExecutionContext::exec(const InstructionList& list) {
	this->_instr = &list;
	while (this->_pc < list.count()){
		list.fetch(this->_pc++)->dispatch(*this);
		if (this->_barrierWait)
			return;
	}
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
	this->_barrierWait = true;
}

ExecResult PtxExecutionContext::result() const{
	return this->_barrierWait ? ExecResult::ThreadSuspended : ExecResult::ThreadExited;
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

typedef struct {
	size_t pc;
	SymbolTable symbols;
	Thread thread;
} thread_launch_data_t;

bool PtxBlockDispatcher::launch(ptx::Function& func, SymbolTable& symbols) {
	const dim3 size(this->_block.size());
	try {
        set_grid_data(symbols, this->_block.grid());
        set_block_data(symbols, this->_block);
		ProtectedStoragePtr shared(std::make_shared<ProtectedStorage>());
		std::vector<thread_launch_data_t> launchData;
		std::vector<thread_launch_data_t> suspendedData;
		for (size_t x=0 ; x<size.x ; ++x) {
			for (size_t y=0 ; y<size.y ; ++y) {
				for (size_t z=0 ; z<size.z ; ++z) {
					Thread thread = this->_block.thread(x,y,z);
					SymbolTable symTab = symbols;
					symTab.setSharedSection(shared);
                    set_thread_data(symTab, x, y, z);
					thread_launch_data_t d;
					d.pc = 0;
					d.symbols = symTab;
					d.thread = thread;
					launchData.push_back(d);
				}
			}
		}
		std::mutex mutex;
		std::vector<std::thread*> threads;
		for (int i=0 ; i<16 ; ++i){
			threads.push_back(new std::thread([&](){
				thread_launch_data_t data;
				bool isEmpty=false;
				while (!isEmpty){
					mutex.lock();
					isEmpty = launchData.empty();
					if (!isEmpty){
						data = launchData[0];
						launchData.erase(launchData.begin());
					}
					mutex.unlock();
					if (!isEmpty) {
						data.symbols.setSharedSection(shared);
						PtxExecutionContext context(this->_device, data.thread, data.symbols);
						context.setProgramCounter(data.pc);
						context.exec(func);
						if (context.result() == ExecResult::ThreadSuspended) {
							thread_launch_data_t d;
							d.pc = context.programCounter();
							d.symbols = data.symbols;
							d.thread = data.thread;
							mutex.lock();
							suspendedData.push_back(d);
							mutex.unlock();
						}
					} else {
						bool waitForSuspended = false;
						mutex.lock();
						if (suspendedData.empty()==false) {
							if (suspendedData.size() == this->_block.threadCount()) {
								std::swap(suspendedData, launchData);
								isEmpty = false;
							} else {
								waitForSuspended = true;
							}
						}
						mutex.unlock();
						if (waitForSuspended) {
							bool allThreadsArrived = false;
							while (!allThreadsArrived){
								mutex.lock();
								allThreadsArrived = suspendedData.empty();
								mutex.unlock();
								std::this_thread::yield();
							}
						}
					}
				}
			 }));
		}
		for (std::thread * th : threads){
			th->join();
			delete th;
		}

	} catch (const std::exception& exc) {
        std::cout << exc.what() << '\n';
		return false;
	}
	return true;
}
