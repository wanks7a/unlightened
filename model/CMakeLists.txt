project(model)

add_executable(main source/main.cpp)

find_path(CATCH_INCLUDE_DIR png.h)

target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CATCH_INCLUDE_DIR}")

target_link_libraries(main unlightened)

