INCLUDEPATH += . \
			   ../.	\
               ../ptx \
               ../ptx/parser \
               ../ptx/semantics \
               ../ptx/runtime \
               ../arch \
               ../ptx/semantics/instructions/memory \
               ../ptx/semantics/instructions/control \
               ../ptx/semantics/instructions/compare \
               ../ptx/semantics/instructions/int \
               ../ptx/semantics/instructions \
               ../drivers/	\
               ../drivers/cuda

HEADERS += ../arch/Device.h \
           ../ptx/Parser.h \
           ../ptx/Tokenizer.h \
           ../drivers/cuda/cuda.h \
           ../drivers/cuda/cudaDefines.h \
           ../drivers/cuda/cudaDriverApi.h \
           ../drivers/cuda/cudaEvent.h \
           ../drivers/cuda/cudaStream.h \
           ../drivers/cuda/cudaThreads.h \
           ../ptx/parser/AbstractParser.h \
           ../ptx/parser/AddParser.h \
           ../ptx/parser/AndParser.h \
           ../ptx/parser/AtomicParser.h \
           ../ptx/parser/BarrierParser.h \
           ../ptx/parser/BranchParser.h \
           ../ptx/parser/CallParser.h \
           ../ptx/parser/ConvertParser.h \
           ../ptx/parser/DirectiveParser.h \
           ../ptx/parser/DivParser.h \
           ../ptx/parser/FunctionParser.h \
           ../ptx/parser/InstructionParser.h \
           ../ptx/parser/LoadParser.h \
           ../ptx/parser/MadParser.h \
           ../ptx/parser/MoveParser.h \
           ../ptx/parser/MulParser.h \
           ../ptx/parser/OrParser.h \
           ../ptx/parser/ParserResult.h \
           ../ptx/parser/ParserUtils.h \
           ../ptx/parser/RemParser.h \
           ../ptx/parser/ReturnExitParser.h \
           ../ptx/parser/SelpParser.h \
           ../ptx/parser/SetpParser.h \
           ../ptx/parser/ShlParser.h \
           ../ptx/parser/StoreParser.h \
           ../ptx/parser/SubParser.h \
           ../ptx/parser/VariableParser.h \
           ../ptx/runtime/PtxExecutionContext.h \
           ../ptx/semantics/Function.h \
           ../ptx/semantics/globals.h \
           ../ptx/semantics/Instruction.h \
           ../ptx/semantics/Semantics.h \
           ../ptx/semantics/Semantics_fwd.h \
           ../ptx/semantics/SymbolTable.h \
           ../ptx/semantics/Variable.h \
           ../ptx/semantics/instructions/FunctionDeclaration.h \
           ../ptx/semantics/instructions/ModuleDirective.h \
           ../ptx/semantics/instructions/VariableDeclaration.h \
           ../ptx/semantics/instructions/compare/CompareInstruction.h \
           ../ptx/semantics/instructions/compare/Selp.h \
           ../ptx/semantics/instructions/compare/Setp.h \
           ../ptx/semantics/instructions/control/Barrier.h \
           ../ptx/semantics/instructions/control/Branch.h \
           ../ptx/semantics/instructions/control/Call.h \
           ../ptx/semantics/instructions/control/ControlInstruction.h \
           ../ptx/semantics/instructions/control/Return.h \
           ../ptx/semantics/instructions/int/Add.h \
           ../ptx/semantics/instructions/int/And.h \
           ../ptx/semantics/instructions/int/Atomic.h \
           ../ptx/semantics/instructions/int/Div.h \
           ../ptx/semantics/instructions/int/Mad.h \
           ../ptx/semantics/instructions/int/Mul.h \
           ../ptx/semantics/instructions/int/Or.h \
           ../ptx/semantics/instructions/int/Rem.h \
           ../ptx/semantics/instructions/int/Shl.h \
           ../ptx/semantics/instructions/int/Sub.h \
           ../ptx/semantics/instructions/memory/Convert.h \
           ../ptx/semantics/instructions/memory/Load.h \
           ../ptx/semantics/instructions/memory/MemoryInstruction.h \
           ../ptx/semantics/instructions/memory/Move.h \
           ../ptx/semantics/instructions/memory/Store.h

SOURCES += ../ptx/Parser.cpp \
           ../ptx/Tokenizer.cpp \
           ../drivers/cuda/cudaDriver_moduleImpl.cpp \
           ../drivers/cuda/cudaDriverApi.cpp \
           ../drivers/cuda/cudaError.cpp \
           ../drivers/cuda/cudaStream.cpp \
           ../ptx/parser/InstructionParser.cpp \
           ../ptx/runtime/PtxExecutionContext.cpp

TEMPLATE = app
TARGET = gemu

CONFIG -= qt
CONFIG += thread c++11
LIBS += -pthread

OBJECTS_DIR = build_tmp

lib {
	TEMPLATE = lib
	TARGET = cuda_lib/cuda
} else {
	TEMPLATE = app
	TARGET = test
	SOURCES += test.cpp \
           testCuda.cpp \
           testParsePtx.cpp \
}
