TARGET=test
CC=g++
CFLAGS+=-std=c++11 -Wall
LIBS+=
LDFLAGS+=-pg
CORE_OBJ=
RM=rm -f
SOURCES=testParsePtx.cpp 	\
		testCuda.cpp		\
		test.cpp			\
		../ptx/runtime/PtxExecutionContext.cpp	\
		../drivers/cuda/cudaDriverApi.cpp

HEADERS=../arch/Device.h	\
		../drivers/cuda/cudaDefines.h	\
		../drivers/cuda/cudaForward.h	\
		../ptx/Parser.h \
		../ptx/Tokenizer.h \
		../ptx/semantics/Semantics.h \
		../ptx/semantics/globals.h \
		../ptx/semantics/Variable.h	\
		../ptx/semantics/Function.h	\
		../ptx/semantics/Instruction.h	\
		../ptx/semantics/instructions/ModuleDirective.h	\
		../ptx/semantics/instructions/VariableDeclaration.h	\
		../ptx/semantics/instructions/FunctionDeclaration.h	\
		../ptx/semantics/instructions/memory/MemoryInstruction.h	\
		../ptx/semantics/instructions/memory/Load.h	\
		../ptx/semantics/instructions/memory/Store.h	\
		../ptx/semantics/instructions/memory/Move.h	\
		../ptx/semantics/instructions/memory/Convert.h	\
		../ptx/semantics/instructions/control/ControlInstruction.h	\
		../ptx/semantics/instructions/control/Return.h	\
		../ptx/parser/ParserResult.h	\
		../ptx/parser/AbstractParser.h	\
		../ptx/parser/ParserUtils.h	\
		../ptx/parser/FunctionParser.h	\
		../ptx/parser/VariableParser.h	\
		../ptx/parser/InstructionParser.h	\
		../ptx/parser/LoadStoreParser.h	\
		../ptx/parser/MoveParser.h	\
		../ptx/parser/ConvertParser.h	\
		../ptx/parser/ReturnExitParser.h	\

INCLUDES=-I../ptx -I../ptx/semantics -I../ptx/parser -I../drivers/cuda -I../drivers -I../

all: test

test: $(CORE_OBJ) $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(LDFLAGS) $(CORE_OBJ) $(SOURCES) -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
