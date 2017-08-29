cmake_minimum_required(VERSION 3.5)

project(gemu)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CUDA_DRIVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/drivers/cuda)
set(PTX_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ptx)

set(SOURCE_FILES ${SOURCE_FILES}
    ${CUDA_DRIVER_DIR}/cudaDriverApi.cpp
    ${CUDA_DRIVER_DIR}/cudaDriver_moduleImpl.cpp
    ${CUDA_DRIVER_DIR}/cudaError.cpp
    ${CUDA_DRIVER_DIR}/cudaStream.cpp
    ${CUDA_DRIVER_DIR}/gemuConfig.cpp
    ${PTX_DIR}/Parser.cpp
    ${PTX_DIR}/Tokenizer.cpp
    ${PTX_DIR}/parser/DebugDirectiveParser.cpp
    ${PTX_DIR}/parser/InstructionParser.cpp
    ${PTX_DIR}/runtime/PtxExecutionContext.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers
    ${CUDA_DRIVER_DIR}
    ${PTX_DIR}
    ${PTX_DIR}/semantics
    ${PTX_DIR}/semantics/instructions
    ${PTX_DIR}/runtime
)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCE_FILES})
set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES OUTPUT_NAME "cuda")

set(TEST_PROJECT_NAME gemu_tests)
set(TEST_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/testCuda.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/testParsePtx.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test/CudaApiLoader.cpp
)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/test)
add_executable(${TEST_PROJECT_NAME} ${TEST_FILES})
target_link_libraries(${TEST_PROJECT_NAME} ${CMAKE_PROJECT_NAME})
