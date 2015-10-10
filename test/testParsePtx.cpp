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
	result = token.tokenize("cvt.rn.f32.u32	%f1, %r1;");
	assert(result.size() == 8);
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

static void test_parser_branch() {
	const std::string source = ".visible .entry kernel_2( .param .u64 kernel_2_param_0, .param .u64 kernel_2_param_1 ){\n"
	".reg .pred 	%p<2>;\n"
	".reg .s32 	%r<5>;\n"
	".reg .s64 	%rd<7>;\n"
	"ld.param.u64 	%rd1, [kernel_2_param_0];\n"
	"ld.param.u64 	%rd2, [kernel_2_param_1];\n"
	"bra 	BB1_2;\n"
	"cvta.to.global.u64 	%rd3, %rd2;\n"
	"ldu.global.u32 	%r2, [%rd3];\n"
	"BB1_2:\n"
	"ret;\n }";
	ptx::ParserResult result = ptx::Parser().parseModule(source);
	assert(result.empty()==false);
	ptx::Function kernel = result.fetch<ptx::FunctionDeclaration>(0)->func();
	assert(kernel.hasLabel("BB1_2"));
	assert(kernel.fetch<ptx::Return>(kernel.instructionIndex("BB1_2")));
	assert(kernel.fetch<ptx::VariableDeclaration>(0));
	assert(kernel.fetch<ptx::VariableDeclaration>(1));
	assert(kernel.fetch<ptx::VariableDeclaration>(2));
	assert(kernel.fetch<ptx::Load>(3));
	assert(kernel.fetch<ptx::Load>(4));
	assert(kernel.fetch<ptx::Branch>(5));
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
	assert(result.empty()==false);
	ptx::Function kernel = result.fetch<ptx::FunctionDeclaration>(0)->func();
	assert(kernel.hasLabel("BB1_2"));
	assert(kernel.fetch<ptx::Return>(kernel.instructionIndex("BB1_2")));
	assert(kernel.fetch<ptx::VariableDeclaration>(0));
	assert(kernel.fetch<ptx::VariableDeclaration>(1));
	assert(kernel.fetch<ptx::VariableDeclaration>(2));
	assert(kernel.fetch<ptx::Load>(3));
	assert(kernel.fetch<ptx::Load>(4));
	assert(kernel.fetch<ptx::Convert>(5));
	assert(kernel.fetch<ptx::Load>(6));
	assert(kernel.fetch<ptx::Move>(7));
	assert(kernel.fetch<ptx::Setp>(8));
	assert(kernel.fetch<ptx::Branch>(9));
	assert(kernel.fetch<ptx::Convert>(10));
	assert(kernel.fetch<ptx::Mul>(11));
	assert(kernel.fetch<ptx::Add>(12));
	assert(kernel.fetch<ptx::Load>(13));
	assert(kernel.fetch<ptx::Shl>(14));
	assert(kernel.fetch<ptx::Store>(15));
	assert(kernel.fetch<ptx::Return>(16));
}

static void test_parser_matrix(){
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry matrix_add(\n"
	".param .u64 matrix_add_param_0,\n"
	".param .u64 matrix_add_param_1,\n"
	".param .u64 matrix_add_param_2,\n"
	".param .u32 matrix_add_param_3,\n"
	".param .u32 matrix_add_param_4 ){\n"
	".reg .pred 	%p<4>;\n"
	".reg .f32 	%f<4>;\n"
	".reg .s32 	%r<6>;\n"
	".reg .s64 	%rd<11>;\n"
	"ld.param.u64 	%rd1, [matrix_add_param_0];\n"
	"ld.param.u64 	%rd2, [matrix_add_param_1];\n"
	"ld.param.u64 	%rd3, [matrix_add_param_2];\n"
	"ld.param.u32 	%r4, [matrix_add_param_3];\n"
	"ld.param.u32 	%r3, [matrix_add_param_4];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"mov.u32 	%r2, %tid.y;\n"
	"setp.lt.u32	%p1, %r2, %r3;\n"
	"setp.lt.u32	%p2, %r1, %r4;\n"
	"and.pred  	%p3, %p1, %p2;\n"
	"@!%p3 bra 	BB0_2;\n"
	"bra.uni 	BB0_1;\n"
	"BB0_1:\n"
	"cvta.to.global.u64 	%rd4, %rd2;\n"
	"mad.lo.s32 	%r5, %r1, %r3, %r2;\n"
	"mul.wide.u32 	%rd5, %r5, 4;\n"
	"add.s64 	%rd6, %rd4, %rd5;\n"
	"cvta.to.global.u64 	%rd7, %rd3;\n"
	"add.s64 	%rd8, %rd7, %rd5;\n"
	"ld.global.f32 	%f1, [%rd8];\n"
	"ld.global.f32 	%f2, [%rd6];\n"
	"add.f32 	%f3, %f2, %f1;\n"
	"cvta.to.global.u64 	%rd9, %rd1;\n"
	"add.s64 	%rd10, %rd9, %rd5;\n"
	"st.global.f32 	[%rd10], %f3;\n"
	"BB0_2:\n"
	"ret;\n"
	"}";
	ptx::ParserResult result = ptx::Parser().parseModule(source);
	assert(result.empty()==false);
}

static void test_parse_sync(){
	const std::string source =
	".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry kernel(\n"
	".param .u64 _Z6kernelPii_param_0,\n"
	".param .u32 _Z6kernelPii_param_1 ) {\n"
	".reg .pred 	%p<5>;\n"
	".reg .s32 	%r<19>;\n"
	".reg .s64 	%rd<3>;\n"
	".shared .u32 _Z6kernelPii$__cuda_local_var_41819_30_non_const_counter;\n"
	"ld.param.u64 	%rd1, [_Z6kernelPii_param_0];\n"
	"ld.param.u32 	%r7, [_Z6kernelPii_param_1];\n"
	"mov.u32 	%r1, %tid.x;\n"
	"setp.ne.s32	%p1, %r1, 0;\n"
	"@%p1 bra 	BB0_2;\n"
	"mov.u32 	%r8, 0;\n"
	"st.shared.u32 	[_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter], %r8;\n"
	"BB0_2:\n"
	"bar.sync 	0;\n"
	"mov.u32 	%r17, 0;\n"
	"mov.u32 	%r18, %r17;\n"
	"mov.u32 	%r15, %r17;\n"
	"setp.lt.s32	%p2, %r7, 1;\n"
	"@%p2 bra 	BB0_4;\n"
	"BB0_3:\n"
	"add.s32 	%r18, %r15, %r18;\n"
	"add.s32 	%r15, %r15, 1;\n"
	"setp.lt.s32	%p3, %r15, %r7;\n"
	"mov.u32 	%r17, %r18;\n"
	"@%p3 bra 	BB0_3;\n"
	"BB0_4:\n"
	"bar.sync 	0;\n"
	"ld.shared.u32 	%r12, [_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter];\n"
	"add.s32 	%r13, %r12, %r17;\n"
	"st.shared.u32 	[_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter], %r13;\n"
	"bar.sync 	0;\n"
	"@%p1 bra 	BB0_6;\n"
	"cvta.to.global.u64 	%rd2, %rd1;\n"
	"ld.shared.u32 	%r14, [_Z6kernelPii$__cuda_local_var_41819_30_non_const_counter];\n"
	"st.global.u32 	[%rd2], %r14;\n"
	"BB0_6:\n"
	"ret;\n"
	"}";
	ptx::ParserResult result = ptx::Parser().parseModule(source);
	assert(result.empty()==false);
}

void test_ptx() {
	std::cout << "testing parser...\n";
	test_variable_parser();
	test_tokenizer();
	test_parser();
	test_parser_branch();
	test_parser_2();
	test_parser_matrix();
	test_parse_sync();
	std::cout << "done.\n";
}
