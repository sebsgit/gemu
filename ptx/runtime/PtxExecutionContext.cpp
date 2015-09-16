#include "runtime/PtxExecutionContext.h"
#include "semantics/Semantics.h"

#include <iostream>

using namespace ptx;
using namespace exec;

void PtxExecutionContext::exec(const InstructionList& list) {
	for (size_t i=0 ; i<list.count() ; ++i)
		list.fetch(i)->dispatch(*this);
}

void PtxExecutionContext::exec(const Instruction& i) {
	std::cout << "!!!EXEC DEFAULT: " << i.toString() << "\n";
	exit(0);
}
void PtxExecutionContext::exec(const Load& load) {
	std::cout << "exec load: " << load.toString() << "\n";
}
void PtxExecutionContext::exec(const Store& store) {
	std::cout << "exec store: " << store.toString() << "\n";
}
void PtxExecutionContext::exec(const Move& move) {
	std::cout << "exec move: " << move.toString() << "\n";
}
void PtxExecutionContext::exec(const Return& r) {
	std::cout << "exec return: " << r.toString() << "\n";
}
void PtxExecutionContext::exec(const Convert& conv) {
	std::cout << "exec conv: " << conv.toString() << "\n";
}
void PtxExecutionContext::exec(const FunctionDeclaration& fdecl) {
	std::cout << "exec func decl: " << fdecl.toString() << "\n";
}
void PtxExecutionContext::exec(const ModuleDirective& d) {
	std::cout << "exec directive: " << d.toString() << "\n";
}
void PtxExecutionContext::exec(const VariableDeclaration& var) {
	std::cout << "exec var decl: " << var.toString() << "\n";
}
