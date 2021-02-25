project(model)

find_package(libpng CONFIG REQUIRED)

find_package(SDL2 CONFIG REQUIRED)
find_package(sdl2-image CONFIG REQUIRED)

find_package(JPEG REQUIRED)
find_path(STB_INCLUDE_DIRS "stb.h")


add_executable(main source/main.cpp
		            include/viewer.h
					source/viewer.cpp
					include/shape_plot.h
					source/shape_plot.cpp
					include/image_helper.h
					source/image_helper.cpp
					include/cat_to_human.h
					source/cat_to_human.cpp
					)

target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include"  ${STB_INCLUDE_DIRS})

target_link_libraries(main PRIVATE SDL2::SDL2 SDL2::SDL2main SDL2::SDL2_image ${JPEG_LIBRARIES} unlightened png)