TARGET=test
CC=g++
CFLAGS+=-std=c++11 -Wall
LIBS+=
LDFLAGS+=
CORE_OBJ=
RM=rm -f
SOURCES=testParsePtx.cpp test.cpp
HEADERS=../ptx/Parser.h \
		../ptx/Tokenizer.h \
		../ptx/semantics/Semantics.h \
		../ptx/semantics/globals.h \
		../ptx/semantics/Variable.h	\
		../ptx/semantics/Function.h	\
		../ptx/semantics/Instruction.h	\
		../ptx/parser/ParserResult.h	\
		../ptx/parser/AbstractParser.h	\
		../ptx/parser/FunctionParser.h	\
		../ptx/parser/VariableParser.h
INCLUDES=-I../ptx -I../ptx/semantics -I../ptx/parser

all: test

test: $(CORE_OBJ) $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LIBS) $(LDFLAGS) $(CORE_OBJ) $(SOURCES) -o $(TARGET)

clean:
	$(RM) $(CORE_OBJ) 2>/dev/null
distclean:
	$(RM) $(CORE_OBJ) 2>/dev/null
	$(RM) $(TARGET) 2>/dev/null
