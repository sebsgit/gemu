#ifndef KERNELDEBUGGERCLI_H
#define KERNELDEBUGGERCLI_H

#ifdef PTX_KERNEL_DEBUG

#include "KernelDebugger.h"

#include <iostream>
#include <string>

namespace ptx {
namespace debug {
	class KernelDebuggerCLI {
	public:
		KernelDebuggerCLI(KernelDebugger& d)
			: _dbg(d)
		{
			this->exec();
		}
	private:
		void exec() {
			const std::string prompt("gemu dbg>");
			while (1) {
				std::cout << prompt;
				std::fflush(stdout);
				auto command = this->getCommand();
				if (command.empty())
					continue;
				if (command[0] == "quit" || command[0] == "q")
					exit(0);
				else if (command[0] == "step" || command[0] == "s") {
					stepBy(command.size() == 1 ? 1 : atoi(command[1].c_str()));
				} else if (command[0] == "print" || command[0] == "p") {
					if (command.size() > 1) {
						const ptx::Variable var = this->_dbg.symbols().variable(command[1]);
						if (var.name().empty())
							std::cout << "\nUndefined symbol: " << command[1] << "\n";
						else {
							std::cout << "\n* -> " << var.toString() << " " << this->_dbg.symbols()[command[1]].printable(var.type(), var.size()) << "\n";
						}
					}
				} else {
					std::cout << "\nUndefined command\n";
				}
			}
		}
		void stepBy(int count) {
			while (count--) {
				auto instr = this->_dbg.step();
				if (!instr)
					break;
				std::cout << '[' << this->_index++ << ']' << "* -> " << instr->toString() << "\n";
			}
		}
		std::vector<std::string> getCommand() const {
			std::vector<std::string> result;
			std::string line;
			std::getline(std::cin, line);
			std::stringstream ss(line);
			std::string tmp;
			while (std::getline(ss, tmp, ' '))
				if (!tmp.empty())
					result.push_back(tmp);
			return result;
		}

	private:
		unsigned long long _index = 0;
		KernelDebugger& _dbg;
	};
}
}

#endif

#endif // KERNELDEBUGGERCLI_H
