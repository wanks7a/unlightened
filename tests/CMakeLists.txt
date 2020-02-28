project(tests)

enable_testing()

add_executable(tests include/tests_objects.h
					 source/main.cpp
		             source/cnn_tests.cpp
					 source/convering_models_tests.cpp)

find_package(GTest CONFIG REQUIRED)
target_link_libraries(tests GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main unlightened)

target_include_directories(unlightened PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_test(AllTestsInMain tests)