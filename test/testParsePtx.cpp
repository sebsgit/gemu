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

std::ostream& operator << (std::ostream& out, const ptx::TokenList& list) {
	for (size_t i=0 ; i<list.size() ; ++i)
		out << list[i] << " ";
	out << "\n";
	return out;
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
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().size() == 8);
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().space() == ptx::AllocSpace::Parameter);
	assert(result.fetch<ptx::VariableDeclaration>(0)->var().type() == ptx::Type::Unsigned);
	tokens.clear();
	tokens << ".reg" << ".s32" << "%p";
	assert(p.parse(tokens, result));
	assert(result.fetch<ptx::VariableDeclaration>(1));
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().name() == "%p");
	assert(result.fetch<ptx::VariableDeclaration>(1)->var().size() == 4);
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
	assert(kernel.parameters().variable(0).size() == 8);
	assert(kernel.parameters().variable(1).name() == "kernel_param_1");
	assert(kernel.parameters().variable(1).type() == ptx::Type::Signed);
	assert(kernel.parameters().variable(1).space() == ptx::AllocSpace::Parameter);
	assert(kernel.parameters().variable(1).size() == 4);
	assert(kernel.empty()==false);
	assert(kernel.fetch<ptx::VariableDeclaration>(0));
	assert(kernel.fetch<ptx::VariableDeclaration>(0)->var().name() == "%r<2>");
	assert(kernel.fetch<ptx::VariableDeclaration>(0)->var().size() == 4);
	assert(kernel.fetch<ptx::VariableDeclaration>(0)->var().space() == ptx::AllocSpace::Register);
	assert(kernel.fetch<ptx::VariableDeclaration>(0)->var().type() == ptx::Type::Signed);
	assert(kernel.fetch<ptx::VariableDeclaration>(1));
	assert(kernel.fetch<ptx::VariableDeclaration>(1)->var().name() == "%rd<3>");
	assert(kernel.fetch<ptx::VariableDeclaration>(1)->var().size() == 8);
	assert(kernel.fetch<ptx::VariableDeclaration>(1)->var().space() == ptx::AllocSpace::Register);
	assert(kernel.fetch<ptx::VariableDeclaration>(1)->var().type() == ptx::Type::Signed);
	assert(kernel.fetch<ptx::Load>(2));
	assert(kernel.fetch<ptx::Load>(2)->cacheMode() == ptx::CacheOperation::CacheAllLevels);
	assert(kernel.fetch<ptx::Load>(2)->space() == ptx::AllocSpace::Parameter);
	assert(kernel.fetch<ptx::Load>(2)->type() == ptx::Type::Unsigned);
	assert(kernel.fetch<ptx::Load>(2)->vectorized() == ptx::VectorType::VecNone);
	assert(kernel.fetch<ptx::Load>(2)->size() == 8);
	assert(kernel.fetch<ptx::Load>(2)->isVolatile() == false);
	assert(kernel.fetch<ptx::Load>(2)->operands().count() == 2);
	assert(kernel.fetch<ptx::Load>(2)->operands()[0].symbol() == "%rd1");
	assert(kernel.fetch<ptx::Load>(2)->operands()[0].isAddressed() == false);
	assert(kernel.fetch<ptx::Load>(2)->operands()[0].offset() == 0);
	assert(kernel.fetch<ptx::Load>(2)->operands()[1].symbol() == "kernel_param_0");
	assert(kernel.fetch<ptx::Load>(2)->operands()[1].isAddressed() == true);
	assert(kernel.fetch<ptx::Load>(2)->operands()[1].offset() == 0);
	assert(kernel.fetch<ptx::Convert>(3));
	assert(kernel.fetch<ptx::Move>(4));
	assert(kernel.fetch<ptx::Move>(4)->cacheMode() == ptx::CacheOperation::CacheAllLevels);
	assert(kernel.fetch<ptx::Move>(4)->space() == ptx::AllocSpace::Register);
	assert(kernel.fetch<ptx::Move>(4)->vectorized() == ptx::VectorType::VecNone);
	assert(kernel.fetch<ptx::Move>(4)->size() == 4);
	assert(kernel.fetch<ptx::Move>(4)->isVolatile() == false);
	assert(kernel.fetch<ptx::Move>(4)->operands().count() == 2);
	assert(kernel.fetch<ptx::Move>(4)->operands()[0].symbol() == "%r1");
	assert(kernel.fetch<ptx::Move>(4)->operands()[0].isAddressed() == false);
	assert(kernel.fetch<ptx::Move>(4)->operands()[0].offset() == 0);
	assert(kernel.fetch<ptx::Move>(4)->operands()[1].symbol() == "5");
	assert(kernel.fetch<ptx::Move>(4)->operands()[1].isAddressed() == false);
	assert(kernel.fetch<ptx::Move>(4)->operands()[1].offset() == 0);
	assert(kernel.fetch<ptx::Store>(5));
	assert(kernel.fetch<ptx::Store>(5)->cacheMode() == ptx::CacheOperation::CacheAllLevels);
	assert(kernel.fetch<ptx::Store>(5)->space() == ptx::AllocSpace::Global);
	assert(kernel.fetch<ptx::Store>(5)->type() == ptx::Type::Unsigned);
	assert(kernel.fetch<ptx::Store>(5)->vectorized() == ptx::VectorType::VecNone);
	assert(kernel.fetch<ptx::Store>(5)->size() == 4);
	assert(kernel.fetch<ptx::Store>(5)->isVolatile() == false);
	assert(kernel.fetch<ptx::Store>(5)->operands().count() == 2);
	assert(kernel.fetch<ptx::Store>(5)->operands()[0].symbol() == "%rd2");
	assert(kernel.fetch<ptx::Store>(5)->operands()[0].isAddressed() == true);
	assert(kernel.fetch<ptx::Store>(5)->operands()[0].offset() == 0);
	assert(kernel.fetch<ptx::Store>(5)->operands()[1].symbol() == "%r1");
	assert(kernel.fetch<ptx::Store>(5)->operands()[1].isAddressed() == false);
	assert(kernel.fetch<ptx::Store>(5)->operands()[1].offset() == 0);
	assert(kernel.last<ptx::Return>());
	assert(kernel.last<ptx::Return>()->isDivergent());
}

static void test_parser_2() {
	const std::string source = ".visible .entry kernel_2( .param .u64 kernel_2_param_0, .param .u64 kernel_2_param_1 ){\n"
	".reg .pred 	%p<2>;\n"
	".reg .s32 	%r<5>;\n"
	".reg .s64 	%rd<7>;\n"
	"ld.param.u64 	%rd1, [kernel_2_param_0];\n"
	"ld.param.u64 	%rd2, [kernel_2_param_1];\n"
	"cvta.to.global.u64 	%rd3, %rd2;\n"
	"ldu.global.u32 	%r2, [%rd3];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.ge.u32	%p1, %r1, %r2;\n"
	"@%p1 bra 	BB1_2;\n"
	"cvta.to.global.u64 	%rd4, %rd1;\n"
	"mul.wide.u32 	%rd5, %r1, 4;\n"
	"add.s64 	%rd6, %rd4, %rd5;\n"
	"ld.global.u32 	%r3, [%rd6];\n"
	"shl.b32 	%r4, %r3, 1;\n"
	"st.global.u32 	[%rd6], %r4;\n"
"BB1_2:\n"
	"ret;\n }";
	ptx::ParserResult result = ptx::Parser().parseModule(source);
	// assert(result.empty()==false);
}

void test_ptx() {
	std::cout << "testing parser...\n";
	test_variable_parser();
	test_tokenizer();
	test_parser();
	test_parser_2();
	std::cout << "done.\n";
}
