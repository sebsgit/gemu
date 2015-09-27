TARGET=test
CC=g++
CFLAGS+=-std=c++11 -Wall
LIBS+=
LDFLAGS+=-pg
CORE_OBJ=
RM=rm -f
LIB_NAME=cuda_lib/libcuda.so.1
PTX_SOURCES=../ptx/runtime/PtxExecutionContext.cpp	\
			../drivers/cuda/cudaDriverApi.cpp	\
			../drivers/cuda/cudaDriver_moduleImpl.cpp
SOURCES=testParsePtx.cpp 	\
		testCuda.cpp		\
		test.cpp			\
		$(PTX_SOURCES)

HEADERS=../arch/Device.h	\
		../drivers/cuda/cudaDefines.h	\
		../drivers/cuda/cuda.h	\
		../drivers/cuda/cudaDriverApi.h	\
		../drivers/cuda/cudaThreads.h	\
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
		../ptx/semantics/instructions/compare/Setp.h	\
		../ptx/semantics/instructions/int/Add.h	\
		../ptx/semantics/instructions/int/Shl.h	\
		../ptx/semantics/instructions/int/Mul.h	\
		../ptx/parser/ParserResult.h	\
		../ptx/parser/AbstractParser.h	\
		../ptx/parser/ParserUtils.h	\
		../ptx/parser/FunctionParser.h	\
		../ptx/parser/VariableParser.h	\
		../ptx/parser/InstructionParser.h	\
		../ptx/parser/LoadParser.h	\
		../ptx/parser/StoreParser.h	\
		../ptx/parser/MoveParser.h	\
		../ptx/parser/ConvertParser.h	\
		../ptx/parser/SetpParser.h	\
		../ptx/parser/MulParser.h	\
		../ptx/parser/AddParser.h	\
		../ptx/parser/ShlParser.h	\
		../ptx/parser/ReturnExitParser.h	\

INCLUDES=-I../ptx -I../ptx/semantics -I../ptx/parser -I../drivers/cuda -I../drivers -I../

all: test

test: $(CORE_OBJ) $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(LDFLAGS) $(CORE_OBJ) $(SOURCES) -o $(TARGET)

lib: $(CORE_OBJ) $(PTX_SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(PTX_SOURCES) -shared -fPIC -o $(LIB_NAME)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
	$(RM) $(LIB_NAME) 2>/dev/null
