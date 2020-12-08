project(tests)

enable_testing()

add_executable(tests include/tests_objects.h
					 source/main.cpp
		             source/cnn_tests.cpp
					 source/converging_models_tests.cpp
					 source/dense_layer_tests.cpp
					 source/generic_functions_tests.cpp
					 source/serialization_tests.cpp
					 source/conv2d_transposed_tests.cpp)

find_package(GTest CONFIG REQUIRED)
target_link_libraries(tests GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main unlightened)

target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_test(AllTestsInMain tests)