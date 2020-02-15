project(tests)

enable_testing()

add_executable(tests source/test.cpp)

find_package(GTest CONFIG REQUIRED)
target_link_libraries(tests GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main unlightened)

add_test(AllTestsInMain tests)