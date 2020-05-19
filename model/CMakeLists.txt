project(model)

find_path(MGL_PATH mgl2/mgl.h)

find_package(libpng CONFIG REQUIRED)

find_package(SDL2 CONFIG REQUIRED)
find_package(sdl2-image CONFIG REQUIRED)

find_library(MGL_LIB mgl)
find_library(MGL_LIB_STATIC mgl-static)

message( ${MGL_LIB} )
message( ${MGL_LIB_STATIC} )

find_package(JPEG REQUIRED)

add_executable(main source/main.cpp
		            include/viewer.h
					source/viewer.cpp
					include/loss_plot.h
					source/loss_plot.cpp
					)

target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${MGL_PATH}")

target_link_libraries(main PRIVATE SDL2::SDL2 SDL2::SDL2main SDL2::SDL2_image ${MGL_LIB} ${JPEG_LIBRARIES} unlightened png) # ${MGL_LIB_STATIC}