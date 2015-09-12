#include "../ptx/Parser.h"

static const std::string test_source = ".version 4.2\n"
		".target sm_20 // a comment\n"
		".address_size 64\n"
		".visible .entry kernel(\n"
		"	.param .u64 kernel_param_0\n"
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

static void test_vars(){
	ptx::Variable bad(".not", ".g62", "badvalue");
	assert(bad.space() == ptx::AllocSpace::undefined);
	assert(bad.size() == 0);

	ptx::Variable var(".param", ".u64", "kernel_param_0");
	assert(var.space() == ptx::AllocSpace::param);
	assert(var.type() == ptx::Type::Unsigned);
	assert(var.size() == 64);
	assert(var.name() == "kernel_param_0");

	var = ptx::Variable(".reg", ".s32", "%p");
	assert(var.space() == ptx::AllocSpace::reg);
	assert(var.type() == ptx::Type::Signed);
	assert(var.size() == 32);
	assert(var.name() == "%p");
}

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
	assert(sublist.size() == 3);
	assert(sublist[0] == ".param");
	assert(sublist[1] == ".u64");
	assert(sublist[2] == "kernel_param_0");
	sublist.removeFirst(2);
	assert(sublist.size() == 1);
	assert(sublist[0] == "kernel_param_0");
}

static void test_parser(){
	
}

void test_ptx() {
	test_vars();
	test_tokenizer();
	test_parser();
}
