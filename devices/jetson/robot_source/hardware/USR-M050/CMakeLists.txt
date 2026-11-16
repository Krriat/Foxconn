cmake_minimum_required(VERSION 3.16)
project(m050_sdk VERSION 1.0.0 LANGUAGES C CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(CTest)
find_package(Threads REQUIRED)

option(M050_BUILD_EXAMPLES "Build read-only C and C++ examples" ON)

add_library(m050 SHARED src/m050.c)
add_library(m050_static STATIC src/m050.c)
add_library(m050::m050 ALIAS m050)
add_library(m050::m050_static ALIAS m050_static)
foreach(target m050 m050_static)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_link_libraries(${target} PUBLIC Threads::Threads)
    set_target_properties(${target} PROPERTIES
        C_STANDARD 99 C_STANDARD_REQUIRED YES C_EXTENSIONS NO
        OUTPUT_NAME m050 POSITION_INDEPENDENT_CODE ON)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endforeach()
set_target_properties(m050 PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 1)

add_library(m050_cpp INTERFACE)
add_library(m050::m050_cpp ALIAS m050_cpp)
target_link_libraries(m050_cpp INTERFACE m050)
target_compile_features(m050_cpp INTERFACE cxx_std_11)

add_executable(m050ctl tools/m050ctl.cpp)
target_link_libraries(m050ctl PRIVATE m050_cpp)
set_target_properties(m050ctl PROPERTIES
    INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(m050ctl PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(M050_BUILD_EXAMPLES)
    add_executable(m050_example_c examples/read_io.c)
    target_link_libraries(m050_example_c PRIVATE m050)
    set_target_properties(m050_example_c PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED YES)
    add_executable(m050_example_cpp examples/read_io.cpp)
    target_link_libraries(m050_example_cpp PRIVATE m050_cpp)
endif()

if(BUILD_TESTING)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME m050_protocol COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_protocol.py
            $<TARGET_FILE:m050> $<TARGET_FILE:m050ctl>
            $<TARGET_FILE:m050_static> ${CMAKE_C_COMPILER}
            ${CMAKE_CURRENT_SOURCE_DIR}/include)
        set_tests_properties(m050_protocol PROPERTIES TIMEOUT 45)
    else()
        message(STATUS "Python3 not found; protocol tests will not be registered")
    endif()
endif()

install(TARGETS m050 m050_static m050_cpp m050ctl EXPORT m050Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT m050Targets NAMESPACE m050::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/m050)
configure_package_config_file(cmake/m050Config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/m050Config.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/m050)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/m050ConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/m050Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/m050ConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/m050)
