#include "../ptx/Parser.h"

static const std::string test_source = ".version 4.2\n"
		".target sm_20 // a comment\n"
		".address_size 64\n"
		".visible .entry kernel(\n"
		"	.param .u64 kernel_param_0,\n"
		"	.param .s32 kernel_param_1\n"
		")\n"
		"{\n"
		"	.reg .s32 	%r<2>; /*another comment\n"
		"spanning multiple lines"
		"	*/.reg .s64 	%rd<3>;\n"
		"\n"
		"\n"
		"	ld.param.u64 	%rd1, [kernel_param_0];\n"
		"	cvta.to.global.u64 	%rd2, %rd1;\n"
		"	mov.u32 	%r1, 5;\n"
		"	st.global.u32 	[%rd2], %r1;\n"
		"	ret;\n"
		"}";

static void test_tokenizer() {
	ptx::Tokenizer token;
	auto result = token.tokenize(test_source);
	assert(result.size() > 0);
	assert(result[0] == ".version");
	assert(result[1] == "4.2");
	assert(result[4] == ".address_size");
	assert(result[6] == ".visible");
	assert(result.peek() == ".version");
	auto sublist = result.sublist("(", ")");
	assert(sublist.size() == 7);
	assert(sublist[0] == ".param");
	assert(sublist[1] == ".u64");
	assert(sublist[2] == "kernel_param_0");
	sublist.removeFirst(2);
	assert(sublist.size() == 5);
	assert(sublist[0] == "kernel_param_0");
}

static void test_variable_parser() {
	ptx::TokenList tokens;
	tokens << ".param" << ".u64" << "kernel_param_0";
	ptx::ParserResult result;
	ptx::parser::VariableParser p;
	assert(p.parse(tokens, result));
	assert(result.fetch<ptx::VariableDeclaration>(0));
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().name() == "kernel_param_0");
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().size() == 64);
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().space() == ptx::AllocSpace::Parameter);
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().type() == ptx::Type::Unsigned);
	tokens.clear();
	tokens << ".reg" << ".s32" << "%p";
	assert(p.parse(tokens, result));
	assert(result.fetch<ptx::VariableDeclaration>(1));
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().name() == "%p");
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().size() == 32);
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().space() == ptx::AllocSpace::Register);
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().type() == ptx::Type::Signed);
}

static void test_parser(){
	ptx::ParserResult result = ptx::Parser().parseModule(test_source);
	assert(result.empty()==false);
	assert(result.fetch<ptx::ModuleDirective>(0)->version() == 4.2f);
	assert(result.fetch<ptx::ModuleDirective>(1)->target() == "sm_20");
	assert(result.fetch<ptx::ModuleDirective>(2)->addressSize() == 64);
	ptx::Function kernel = result.fetch<ptx::FunctionDeclaration>(3)->func();
	assert(kernel.name() == "kernel");
	assert(kernel.parameters().size() == 2);
	assert(kernel.parameters().variable(0).name() == "kernel_param_0");
	assert(kernel.parameters().variable(0).type() == ptx::Type::Unsigned);
	assert(kernel.parameters().variable(0).space() == ptx::AllocSpace::Parameter);
	assert(kernel.parameters().variable(0).size() == 64);
	assert(kernel.parameters().variable(1).name() == "kernel_param_1");
	assert(kernel.parameters().variable(1).type() == ptx::Type::Signed);
	assert(kernel.parameters().variable(1).space() == ptx::AllocSpace::Parameter);
	assert(kernel.parameters().variable(1).size() == 32);
}

void test_ptx() {
	test_variable_parser();
	test_tokenizer();
	test_parser();
}
