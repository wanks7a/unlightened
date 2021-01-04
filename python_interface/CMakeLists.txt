add_library(
    python_interface SHARED
    python_api.h 
    python_api.cpp
)

target_link_libraries(python_interface unlightened)