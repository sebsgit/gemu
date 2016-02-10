#include "runtime/PtxExecutionContext.h"
#include "semantics/Semantics.h"
#include "../drivers/cuda/cudaDriverApi.h"
#include "../drivers/cuda/gemuConfig.h"
#include <thread>
#include <mutex>

using namespace ptx;
using namespace exec;

#ifdef PTX_KERNEL_DEBUG
#include "debug/KernelDebugger.h"
debug::KernelDebugger* PtxExecutionContext::_debugger = nullptr;
#endif

void SymbolStorage::set(const std::string& name, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
	if (it != _data.end()) {
		it->data = storage;
	}
}
bool SymbolStorage::setIfExists(const std::string& name, const param_storage_t& storage) {
    auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
    if (it != _data.end()) {
        it->data = storage;
        return true;
    }
    return false;
}
bool SymbolStorage::getIfExists(const std::string& name, param_storage_t& storage) const {
    auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
    if (it != _data.end()) {
        storage = it->data;
        return true;
    }
    return false;
}

bool SymbolStorage::getIfExists(const std::string& name, ptx::Variable& var) const {
    auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == name;});
    if (it != _data.end()) {
        var = it->var;
        return true;
    }
    return false;
}

unsigned long long SymbolStorage::address(const std::string& name) const{
	for (size_t i=0 ; i<this->_data.size() ; ++i){
		if (this->_data[i].var.name() == name) {
			return reinterpret_cast<unsigned long long>(&this->_data[i].data);
		}
	}
	return 0;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

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

#pragma GCC diagnostic pop

void SymbolStorage::set(const ptx::Variable& var, const param_storage_t& storage) {
	auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == var.name();});
	if (it != _data.end()) {
		it->data = storage;
	} else {
		_data.push_back(entry_t(var, storage));
	}
}
void SymbolStorage::declare(const Variable &var){
    auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == var.name();});
    if (it == _data.end())
        _data.push_back(entry_t(var, param_storage_t()));
}

bool SymbolStorage::setIfExists(const ptx::Variable& var, const param_storage_t& storage) {
    auto it = std::find_if(_data.begin(), _data.end(), [&](const entry_t& d){ return d.var.name() == var.name();});
    if (it != _data.end()) {
        it->data = storage;
        return true;
    }
    return false;
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
#ifdef PTX_KERNEL_DEBUG
    if (PtxExecutionContext::_debugger)
        return this->exec_debug(list);
#endif
	this->_instr = &list;
	while (this->_pc < list.count()){
		list.fetch(this->_pc++)->dispatch(*this);
		if (this->_barrierWait)
			return;
	}
	this->_instr = nullptr;
}

#ifdef PTX_KERNEL_DEBUG
void PtxExecutionContext::exec_debug(const InstructionList& list) {
    PtxExecutionContext::_debugger->exec(this, list);
}
#endif

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
    PTX_UNUSED(r);
	if (this->_instr)
		this->_pc = this->_instr->count();
}
void PtxExecutionContext::exec(const FunctionDeclaration& fdecl) {
	// std::cout << "exec func decl: " << fdecl.toString() << "\n";
    PTX_UNUSED(fdecl);
}
void PtxExecutionContext::exec(const ModuleDirective& d) {
	// std::cout << "exec directive: " << d.toString() << "\n";
    PTX_UNUSED(d);
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
    PTX_UNUSED(barrier);
	this->_barrierWait = true;
}

void PtxExecutionContext::exec(const Call &call) {
    ptx::Function func = _driverContext->function(call.target());
    SymbolTable table;
    table.setSharedSection(this->_symbols.sharedSection());
    table.set(func.returnVariable(), param_storage_t());
    for (size_t i=0 ; i<call.parameterCount() ; ++i) {
        const auto callParam = this->_symbols.get(call.parameter(i));
        table.set(func.parameters()[i], callParam);
    }
    PtxExecutionContext context(this->_device, this->_thread, table);
    context.exec(func);
    const auto result = table.get(func.returnVariable().name());
    this->_symbols.set(func.returnVariable().renamed(call.result()), result);
}

ThreadExecResult PtxExecutionContext::result() const{
    return this->_barrierWait ? ThreadExecResult::ThreadSuspended : ThreadExecResult::ThreadExited;
}

using namespace gemu;
using namespace cuda;

typedef struct {
    size_t pc;
    SymbolTable symbols;
    Thread thread;
} thread_launch_data_t;

struct PtxBlockDispatcher::Data {
    std::mutex mutex;
    std::vector<std::thread*> threads;
    ProtectedStoragePtr shared = std::make_shared<ProtectedStorage>();
    std::vector<thread_launch_data_t> launchData;
    std::vector<thread_launch_data_t> suspendedData;
};

PtxBlockDispatcher::PtxBlockDispatcher(gemu::Device& device, gemu::cuda::ThreadBlock& block)
:_device(device)
,_block(block)
,_data(new Data)
{

}

PtxBlockDispatcher::~PtxBlockDispatcher(){
    delete _data;
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

void PtxBlockDispatcher::launch(ptx::Function& func, SymbolTable& symbols) {
	const dim3 size(this->_block.size());
    this->_result = BlockExecResult::BlockRunning;
	try {
        set_grid_data(symbols, this->_block.grid());
        set_block_data(symbols, this->_block);
		for (size_t x=0 ; x<size.x ; ++x) {
			for (size_t y=0 ; y<size.y ; ++y) {
				for (size_t z=0 ; z<size.z ; ++z) {
					Thread thread = this->_block.thread(x,y,z);
					SymbolTable symTab = symbols;
                    symTab.setSharedSection(this->_data->shared);
                    set_thread_data(symTab, x, y, z);
					thread_launch_data_t d;
					d.pc = 0;
					d.symbols = symTab;
					d.thread = thread;
                    this->_data->launchData.push_back(d);
				}
			}
		}
        int maxThreads = gemu::config::defaultDevice.maxRuntimeThreads;
        #ifdef PTX_KERNEL_DEBUG
        maxThreads = 1;
        #endif
        for (int i=0 ; i<maxThreads ; ++i){
            this->_data->threads.push_back(new std::thread([&](){
				thread_launch_data_t data;
				bool isEmpty=false;
				while (!isEmpty){
                    this->_data->mutex.lock();
                    isEmpty = this->_data->launchData.empty();
					if (!isEmpty){
                        data = this->_data->launchData[0];
                        this->_data->launchData.erase(this->_data->launchData.begin());
					}
                    this->_data->mutex.unlock();
					if (!isEmpty) {
                        data.symbols.setSharedSection(this->_data->shared);
						PtxExecutionContext context(this->_device, data.thread, data.symbols);
						context.setProgramCounter(data.pc);
						context.exec(func);
                        if (context.result() == ThreadExecResult::ThreadSuspended) {
							thread_launch_data_t d;
							d.pc = context.programCounter();
							d.symbols = data.symbols;
							d.thread = data.thread;
                            this->_data->mutex.lock();
                            this->_data->suspendedData.push_back(d);
                            this->_data->mutex.unlock();
						}
					} else {
						bool waitForSuspended = false;
                        this->_data->mutex.lock();
                        if (this->_data->suspendedData.empty()==false) {
                            if (this->_data->suspendedData.size() == this->_block.threadCount()) {
                                std::swap(this->_data->suspendedData, this->_data->launchData);
								isEmpty = false;
							} else {
								waitForSuspended = true;
							}
						}
                        this->_data->mutex.unlock();
						if (waitForSuspended) {
							bool allThreadsArrived = false;
							while (!allThreadsArrived){
                                this->_data->mutex.lock();
                                allThreadsArrived = this->_data->suspendedData.empty();
                                this->_data->mutex.unlock();
								std::this_thread::yield();
							}
						}
					}
				}
			 }));
		}
	} catch (const std::exception& exc) {
        std::cout << exc.what() << '\n';
        this->_result = BlockExecResult::BlockError;
	}
}

void PtxBlockDispatcher::synchronize(){
    for (std::thread * th : this->_data->threads){
        th->join();
        delete th;
    }
    this->_data->threads.clear();
    if (this->_result == BlockExecResult::BlockRunning)
        this->_result = BlockExecResult::BlockOk;
}
