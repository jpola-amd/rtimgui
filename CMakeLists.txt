cmake_minimum_required(VERSION 3.26)
project(hiprt-trace CXX)

set(GPU_ARCH gfx1100)

if(WIN32)
    set(HIP_PATH "C:/rocm/5.7")
    set(HIPRT_PATH "E:/HIPRT")
else()
    set(HIP_PATH "/opt/rocm")
endif()

set(HIPRT_INCLUDE ${HIPRT_PATH})
set(HIPRT_BINDIR ${HIPRT_PATH}/dist/bin/$<CONFIG>)

find_package(OpenGL 3 REQUIRED)


include(FetchContent)
FetchContent_Declare(stl_reader
        GIT_REPOSITORY https://github.com/sreiter/stl_reader.git
        GIT_TAG master)

FetchContent_Declare(stb_image
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)
FetchContent_Declare(tinyobj
    GIT_REPOSITORY https://github.com/tinyobjloader/tinyobjloader.git
    GIT_TAG release
)


FetchContent_Declare(
	glad
	GIT_REPOSITORY https://github.com/Dav1dde/glad.git
	GIT_TAG v0.1.36
	)
	
set(GLAD_PROFILE "core"	CACHE STRING "OpenGL profile")

set(GLAD_GENERATOR "c"	CACHE STRING "Language to generate the binding for")

#FetchContent_MakeAvailable(glad)



set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw 
    GIT_REPOSITORY https://github.com/glfw/glfw.git 
    GIT_TAG 3.4
)

FetchContent_MakeAvailable(
    stl_reader
    stb_image
    tinyobj
    glfw
    glad
)

FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.4
)
FetchContent_GetProperties(imgui SOURCE_DIR IMGUI_DIR)
if (NOT imgui_POPULATED)
    FetchContent_Populate(imgui)
    set(IMGUI_INCLUDE_DIR ${imgui_SOURCE_DIR}/)
    file(GLOB IMGUI_SOURCES ${imgui_SOURCE_DIR}/*.cpp)
    file(GLOB IMGUI_HEADERS ${imgui_SOURCE_DIR}/*.h)
    add_library(imgui STATIC ${IMGUI_SOURCES} ${IMGUI_SOURCES} ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)
    #]
    #[link_imgui Link ImGUI
    target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIR} ${GLAD_INCLUDE_DIR})
    target_link_libraries(imgui ${OPENGL_LIBRARIES} glfw)
endif () 

#add_library(
 #   imgui STATIC
 #   ${imgui_SOURCE_DIR}/imgui.cpp
 #   ${imgui_SOURCE_DIR}/imgui_draw.cpp
 #   ${imgui_SOURCE_DIR}/imgui_widgets.cpp
 #   ${imgui_SOURCE_DIR}/imgui_tables.cpp
 #   ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
 #   ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
 #   )

#target_include_directories(imgui PUBLIC ${IMGUI_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${GLFW_INCLUDE_DIR})
#target_link_libraries(imgui ${OPENGL_LIBRARIES} glfw)
#target_include_directories(imgui INTERFACE ${imgui_SOURCE_DIR})


message(STATUS "STL READER: ${stl_reader_SOURCE_DIR}")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_DEBUG_POSTFIX "-d")

message(STATUS "OUTPUT: ${PROJECT_BINARY_DIR}")
add_subdirectory(kernels) 
add_subdirectory(src)

#add_dependencies (HIPRT-AO imgui glfw)
add_dependencies (HIPRT-AO HIPRT-TraceKernels)

