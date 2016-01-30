#include "../ptx/Parser.h"
#include "../ptx/parser/CallParser.h"
#include "../ptx/semantics/instructions/control/Call.h"
#include "../ptx/parser/VariableParser.h"
#include <cassert>

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
    for (const auto& t : list)
        out << t << " ";
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
    result = token.tokenize("st.param.b32	[func_retval0+0], %r3;");
    assert(result.size() == 11);
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
    ptx::Function kernel = result.fetch<ptx::FunctionDeclaration>(3)->func();
    assert(kernel.name() == "matrix_add");
    assert(kernel.parameters().size() == 5);
    assert(kernel.hasLabel("BB0_1"));
    assert(kernel.fetch<ptx::Convert>(kernel.instructionIndex("BB0_1")));
    assert(kernel.hasLabel("BB0_2"));
    assert(kernel.fetch<ptx::Return>(kernel.instructionIndex("BB0_2")));
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

static void test_parse_atomic(){
    const std::string source =
    ".version 4.2\n"
	".target sm_20\n"
	".address_size 64\n"
	".visible .entry _Z6kernelPi("
	"	.param .u64 _Z6kernelPi_param_0"
	")"
	"{"
	"	.reg .pred 	%p<3>;"
	"	.reg .s32 	%r<5>;"
	"	.reg .s64 	%rd<4>;"
	"	.shared .u32 _Z6kernelPi$__cuda_local_var_41819_30_non_const_count;"
	""
	"	ld.param.u64 	%rd1, [_Z6kernelPi_param_0];"
	"	mov.u32 	%r1, %tid.x;"
	"	setp.ne.s32	%p2, %r1, 0;"
	"	@%p2 bra 	BB0_2;"
	""
	"	mov.u32 	%r2, 0;"
	"	st.shared.u32 	[_Z6kernelPi$__cuda_local_var_41819_30_non_const_count], %r2;"
	""
	"BB0_2:"
	"	setp.eq.s32	%p1, %r1, 0;"
	"	bar.sync 	0;"
	"	mov.u64 	%rd2, _Z6kernelPi$__cuda_local_var_41819_30_non_const_count;"
	"	atom.shared.add.u32 	%r3, [%rd2], 1;"
	"	bar.sync 	0;"
	"	@!%p1 bra 	BB0_4;"
	"	bra.uni 	BB0_3;"
    ""
	"BB0_3:"
	"	cvta.to.global.u64 	%rd3, %rd1;"
	"	ld.shared.u32 	%r4, [_Z6kernelPi$__cuda_local_var_41819_30_non_const_count];"
	"	st.global.u32 	[%rd3], %r4;"
	""
	"BB0_4:"
	"	ret;"
	"}";
	ptx::ParserResult result = ptx::Parser().parseModule(source);
	assert(result.empty()==false);
}

static void test_parse_two_functions() {
    const std::string source = ".version 4.2\n"
   ".target sm_20\n"
   ".address_size 64\n"
   ".visible .func  (.param .b32 func_retval0) _Z4funcii("
   "	.param .b32 _Z4funcii_param_0,"
   "	.param .b32 _Z4funcii_param_1"
   ")"
   "{"
   "	.reg .s32 	%r<4>;"
   ""
   ""
   "	ld.param.u32 	%r1, [_Z4funcii_param_0];"
   "	ld.param.u32 	%r2, [_Z4funcii_param_1];"
   "	add.s32 	%r3, %r2, %r1;"
   "	st.param.b32	[func_retval0+0], %r3;"
   "	ret;"
   "}"
   ".visible .entry _Z6kernelPiii("
   "	.param .u64 _Z6kernelPiii_param_0,"
   "	.param .u32 _Z6kernelPiii_param_1,"
   "	.param .u32 _Z6kernelPiii_param_2"
   ")"
   "{"
   "	.reg .s32 	%r<4>;"
   "	.reg .s64 	%rd<3>;"
   ""
   ""
   "	ld.param.u64 	%rd1, [_Z6kernelPiii_param_0];"
   "	ld.param.u32 	%r1, [_Z6kernelPiii_param_1];"
   "	ld.param.u32 	%r2, [_Z6kernelPiii_param_2];"
   "	cvta.to.global.u64 	%rd2, %rd1;"
   "	{"
   "	.reg .b32 temp_param_reg;"
   "	.param .b32 param0;"
   "	st.param.b32	[param0+0], %r1;"
   "	.param .b32 param1;"
   "	st.param.b32	[param1+0], %r2;"
   "	.param .b32 retval0;"
   "	call.uni (retval0), "
   "	_Z4funcii, "
   "	("
   "	param0, "
   "	param1"
   "	);"
   "	ld.param.b32	%r3, [retval0+0];"
   "	}"
   "	st.global.u32 	[%rd2], %r3;"
   "	ret;"
   "}";
    ptx::ParserResult result = ptx::Parser().parseModule(source);
    assert(result.empty()==false);
}

static void test_call_parser() {
    const std::string source =    "	call.uni (retval0), "
                                  "	_Z4funcii, "
                                  "	("
                                  "	param0, "
                                  "	param1"
                                  "	);";
    ptx::parser::CallParser parser;
    ptx::ParserResult result;
    auto tokens = ptx::Tokenizer().tokenize(source);
    assert(parser.parse(tokens, result));
    auto call = result.fetch<ptx::Call>(0);
    assert(call);
    assert(call->isDivergent() == false);
    assert(call->target() == "_Z4funcii");
    assert(call->result() == "retval0");
    assert(call->parameterCount() == 2);
    assert(call->parameter(0) == "param0");
    assert(call->parameter(1) == "param1");
}

static void test_parse_3dgrid() {
    const std::string test_source =
        "//\n"
        "// Generated by NVIDIA NVVM Compiler\n"
        "//\n"
        "// Compiler Build ID: CL-19324607\n"
        "// Cuda compilation tools, release 7.0, V7.0.27\n"
        "// Based on LLVM 3.4svn\n"
        "//\n"
        "\n"
        ".version 4.2\n"
        ".target sm_20\n"
        ".address_size 64\n"
        "\n"
        "	// .globl	_Z6kernelPjiii\n"
        "\n"
        ".visible .entry kernel(\n"
        "	.param .u64 _Z6kernelPjiii_param_0,\n"
        "	.param .u32 _Z6kernelPjiii_param_1,\n"
        "	.param .u32 _Z6kernelPjiii_param_2,\n"
        "	.param .u32 _Z6kernelPjiii_param_3\n"
        ")\n"
        "{\n"
        "	.reg .pred 	%p<2>;\n"
        "	.reg .s32 	%r<19>;\n"
        "	.reg .s64 	%rd<5>;\n"
        "\n"
        "\n"
        "	ld.param.u64 	%rd1, [_Z6kernelPjiii_param_0];\n"
        "	ld.param.u32 	%r4, [_Z6kernelPjiii_param_1];\n"
        "	ld.param.u32 	%r5, [_Z6kernelPjiii_param_2];\n"
        "	ld.param.u32 	%r6, [_Z6kernelPjiii_param_3];\n"
        "	mov.u32 	%r7, %tid.x;\n"
        "	mov.u32 	%r8, %ntid.y;\n"
        "	mov.u32 	%r9, %tid.z;\n"
        "	mov.u32 	%r10, %tid.y;\n"
        "	mad.lo.s32 	%r11, %r8, %r9, %r10;\n"
        "	mov.u32 	%r12, %ntid.x;\n"
        "	mad.lo.s32 	%r1, %r11, %r12, %r7;\n"
        "	mul.lo.s32 	%r2, %r5, %r4;\n"
        "	div.s32 	%r3, %r1, %r2;\n"
        "	setp.ge.s32	%p1, %r3, %r6;\n"
        "	@%p1 bra 	BB0_2;\n"
        "\n"
        "	cvta.to.global.u64 	%rd2, %rd1;\n"
        "	mul.lo.s32 	%r13, %r3, %r2;\n"
        "	sub.s32 	%r14, %r1, %r13;\n"
        "	rem.s32 	%r15, %r14, %r4;\n"
        "	div.s32 	%r16, %r14, %r5;\n"
        "	mad.lo.s32 	%r17, %r3, %r5, %r16;\n"
        "	mad.lo.s32 	%r18, %r17, %r4, %r15;\n"
        "	mul.wide.s32 	%rd3, %r18, 4;\n"
        "	add.s64 	%rd4, %rd2, %rd3;\n"
        "	st.global.u32 	[%rd4], %r1;\n"
        "\n"
        "BB0_2:\n"
        "	ret;\n"
        "}\n";
    ptx::ParserResult result = ptx::Parser().parseModule(test_source);
    assert(result.empty()==false);
}

static void test_with_short() {
    const std::string source =
        ".visible .entry kernel(\n"
        "	.param .u64 _Z6kernelPsS_S__param_0,\n"
        "	.param .u64 _Z6kernelPsS_S__param_1,\n"
        "	.param .u64 _Z6kernelPsS_S__param_2\n"
        ")\n"
        "{\n"
        "	.reg .b32 	%r<4>;\n"
        "	.reg .b64 	%rd<7>;\n"
        "\n"
        "\n"
        "	ld.param.u64 	%rd1, [_Z6kernelPsS_S__param_0];\n"
        "	ld.param.u64 	%rd2, [_Z6kernelPsS_S__param_1];\n"
        "	ld.param.u64 	%rd3, [_Z6kernelPsS_S__param_2];\n"
        "	cvta.to.global.u64 	%rd4, %rd3;\n"
        "	cvta.to.global.u64 	%rd5, %rd2;\n"
        "	cvta.to.global.u64 	%rd6, %rd1;\n"
        "	ld.global.u16 	%r1, [%rd6];\n"
        "	ld.global.u16 	%r2, [%rd5];\n"
        "	add.s32 	%r3, %r2, %r1;\n"
        "	st.global.u16 	[%rd4], %r3;\n"
        "	ret;\n"
        "}\n"
        "\n"
        "\n"
    ;
    ptx::ParserResult result = ptx::Parser().parseModule(source);
    assert(result.empty()==false);
    ptx::Function kernel = result.fetch<ptx::FunctionDeclaration>(0)->func();
    assert(kernel.fetch<ptx::VariableDeclaration>(0));
    assert(kernel.fetch<ptx::VariableDeclaration>(1));

    assert(kernel.fetch<ptx::Load>(2));
    assert(kernel.fetch<ptx::Load>(3));
    assert(kernel.fetch<ptx::Load>(4));

    assert(kernel.fetch<ptx::Convert>(5));
    assert(kernel.fetch<ptx::Convert>(6));
    assert(kernel.fetch<ptx::Convert>(7));

    assert(kernel.fetch<ptx::Load>(8));
    assert(kernel.fetch<ptx::Load>(9));

    assert(kernel.fetch<ptx::Add>(10));
    assert(kernel.fetch<ptx::Store>(11));
    assert(kernel.fetch<ptx::Return>(12));
}

static void test_with_global() {
    const std::string source =
        ".version 4.3\n"
        ".target sm_20\n"
        ".address_size 64\n"
        "\n"
        "	// .globl	_Z10kernel_seti\n"
        ".global .align 4 .u32 globalValue;\n"
        "\n"
        ".visible .entry _Z10kernel_seti(\n"
        "	.param .u32 _Z10kernel_seti_param_0\n"
        ")\n"
        "{\n"
        "	.reg .b32 	%r<2>;\n"
        "\n"
        "\n"
        "	ld.param.u32 	%r1, [_Z10kernel_seti_param_0];\n"
        "	st.global.u32 	[globalValue], %r1;\n"
        "	ret;\n"
        "}\n"
        "\n"
        "	// .globl	_Z10kernel_getPi\n"
        ".visible .entry _Z10kernel_getPi(\n"
        "	.param .u64 _Z10kernel_getPi_param_0\n"
        ")\n"
        "{\n"
        "	.reg .b32 	%r<2>;\n"
        "	.reg .b64 	%rd<3>;\n"
        "\n"
        "\n"
        "	ld.param.u64 	%rd1, [_Z10kernel_getPi_param_0];\n"
        "	cvta.to.global.u64 	%rd2, %rd1;\n"
        "	ldu.global.u32 	%r1, [globalValue];\n"
        "	st.global.u32 	[%rd2], %r1;\n"
        "	ret;\n"
        "}\n";
    ptx::ParserResult result = ptx::Parser().parseModule(source);
    assert(result.empty()==false);
}

static void test_with_file_and_loc() {
	const std::string source = "//\n"
		"// Generated by NVIDIA NVVM Compiler\n"
		"// Compiler built on Thu Jul 18 02:37:37 2013 (1374107857)\n"
		"// Cuda compilation tools, release 5.5, V5.5.0\n"
		"//\n"
		"\n"
		".version 3.2\n"
		".target sm_20\n"
		".address_size 64\n"
		"\n"
		"	.file	1 \"/home/seb/gitrepos/gemu/test/cases/test.cu\", 1454142084, 254\n"
		"\n"
		".visible .entry _Z6kernelPjiii(\n"
		"	.param .u64 _Z6kernelPjiii_param_0,\n"
		"	.param .u32 _Z6kernelPjiii_param_1,\n"
		"	.param .u32 _Z6kernelPjiii_param_2,\n"
		"	.param .u32 _Z6kernelPjiii_param_3\n"
		")\n"
		"{\n"
		"	.reg .pred 	%p<6>;\n"
		"	.reg .s32 	%r<9>;\n"
		"	.reg .s64 	%rd<5>;\n"
		"\n"
		"\n"
		"	ld.param.u64 	%rd2, [_Z6kernelPjiii_param_0];\n"
		"	ld.param.u32 	%r4, [_Z6kernelPjiii_param_1];\n"
		"	ld.param.u32 	%r5, [_Z6kernelPjiii_param_2];\n"
		"	ld.param.u32 	%r6, [_Z6kernelPjiii_param_3];\n"
		"	cvta.to.global.u64 	%rd1, %rd2;\n"
		"	.loc 1 2 1\n"
		"	mov.u32 	%r1, %tid.x;\n"
		"	setp.lt.u32	%p1, %r1, %r4;\n"
		"	.loc 1 3 1\n"
		"	mov.u32 	%r2, %tid.y;\n"
		"	setp.lt.u32	%p2, %r2, %r5;\n"
		"	.loc 1 2 1\n"
		"	and.pred  	%p3, %p1, %p2;\n"
		"	.loc 1 4 1\n"
		"	mov.u32 	%r3, %tid.z;\n"
		"	setp.lt.u32	%p4, %r3, %r6;\n"
		"	.loc 1 2 1\n"
		"	and.pred  	%p5, %p3, %p4;\n"
		"	@!%p5 bra 	BB0_2;\n"
		"	bra.uni 	BB0_1;\n"
		"\n"
		"BB0_1:\n"
		"	mad.lo.s32 	%r7, %r3, %r5, %r2;\n"
		"	.loc 1 5 1\n"
		"	mad.lo.s32 	%r8, %r7, %r4, %r1;\n"
		"	mul.wide.u32 	%rd3, %r8, 4;\n"
		"	add.s64 	%rd4, %rd1, %rd3;\n"
		"	st.global.u32 	[%rd4], %r8;\n"
		"\n"
		"BB0_2:\n"
		"	.loc 1 9 2\n"
		"	ret;\n"
		"}\n";
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
    test_parse_atomic();
    test_parse_two_functions();
    test_call_parser();
    test_parse_3dgrid();
    test_with_short();
    test_with_global();
    test_with_file_and_loc();
	std::cout << "done.\n";
}
