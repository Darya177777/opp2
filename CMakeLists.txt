cmake_minimum_required(VERSION 3.20)
project(t C)

set(CMAKE_C_STANDARD 99)
find_package(OpenMP REQUIRED)
link_libraries(m)


add_executable(t main.c)
target_link_libraries(t PRIVATE "${OpenMP_C_FLAGS}")
target_compile_options(t PRIVATE "${OpenMP_C_FLAGS}")

