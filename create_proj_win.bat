IF exist build ( echo build exists ) ELSE ( mkdir build && echo build created)
cd build
cmake ..  "-DCMAKE_TOOLCHAIN_FILE=D:\vcpkg\scripts\buildsystems\vcpkg.cmake" "-DGTest_DIR=D:\vcpkg\packages\gtest_x64-windows\share\gtest"