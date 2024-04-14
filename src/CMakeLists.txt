
project(HIPRT-AO)
set(target HIPRT-AO)


set(sources
    main.cpp
    assert.h
    TriangleMesh.h
    TriangleMesh.cpp
    MeshReader.h
    MeshReader.cpp
    Quaternion.h
    ImageWriter.h 
    ImageWriter.cpp 
    Geometry.h
    Geometry.cpp 
    Scene.h
    Scene.cpp
    RenderCases.h
    Aabb.h 
    DisplayWindow.h
    DisplayWindow.cpp
    )

#select debug or release 
SET(HIPRT_DLL  $<IF:$<CONFIG:Debug>,${HIPRT_BINDIR}/hiprt0200364D.dll,${HIPRT_BINDIR}/hiprt0200364.dll>)
set(HIPRT_LIB  $<IF:$<CONFIG:Debug>,${HIPRT_BINDIR}/hiprt0200364D.lib,${HIPRT_BINDIR}/hiprt0200364.lib>)

add_executable(${target} ${sources})
set_property(TARGET ${target} PROPERTY CXX_STANDARD 20)
target_compile_definitions(${target} PRIVATE "__HIP_PLATFORM_AMD__")
target_include_directories(${target} PRIVATE ${HIP_PATH}/include)
target_include_directories(${target} PRIVATE ${HIPRT_INCLUDE})
target_include_directories(${target} PRIVATE ${stl_reader_SOURCE_DIR})
target_include_directories(${target} PRIVATE ${stb_image_SOURCE_DIR})
target_include_directories(${target} PRIVATE ${tinyobj_SOURCE_DIR})
target_compile_definitions(${target} PUBLIC TINYOBJLOADER_IMPLEMENTATION)
target_compile_definitions(${target} PUBLIC STB_IMAGE_WRITE_IMPLEMENTATION)
target_include_directories(${target} PRIVATE ${glfw_SOURCE_DIR})
target_include_directories(${target} PRIVATE ${imgui_SOURCE_DIR})





if(WIN32)
    target_link_libraries(${target} PRIVATE ${HIP_PATH}/lib/amdhip64.lib)
    target_link_libraries(${target} PRIVATE ${HIPRT_LIB})
    target_link_libraries(${target} PRIVATE glfw)
    target_link_libraries(${target} PRIVATE imgui)
    target_link_libraries(${target} PRIVATE glad)

   
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIPRT_DLL} ${CMAKE_BINARY_DIR}/$<CONFIG>/)

    # Copy dependencies
    #HIP dependencies
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIP_PATH}/bin/amd_comgr0507.dll ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIP_PATH}/bin/hiprtc0507.dll ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIP_PATH}/bin/hiprtc-builtins0507.dll ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    #HIPRT 
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIP_PATH}/bin/hiprtc-builtins0507.dll ${CMAKE_BINARY_DIR}/$<CONFIG>/)       
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIPRT_BINDIR}/hiprt02003_5.7_amd_lib_win.bc ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIPRT_BINDIR}/oro_compiled_kernels.hipfb ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    add_custom_command(TARGET ${target} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${HIPRT_BINDIR}/hiprt02003_5.7_amd.hipfb ${CMAKE_BINARY_DIR}/$<CONFIG>/)
    
else()
    target_link_libraries(${target} PRIVATE ${HIP_PATH}/lib/libamdhip64.so)
    # make output path consistent between MSVC and GCC compilation
    add_custom_command(TARGET ${target}
        POST_BUILD
        COMMAND mv ${RUNTIME_OUTPUT_DIRECTORY}/../${target}${CMAKE_DEBUG_POSTFIX} ${RUNTIME_OUTPUT_DIRECTORY}
    )
endif()
# Set the output directory and debug postfix for TestApp
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY})

set_target_properties(${target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${RUNTIME_OUTPUT_DIRECTORY}
    DEBUG_POSTFIX ${CMAKE_DEBUG_POSTFIX}
)

