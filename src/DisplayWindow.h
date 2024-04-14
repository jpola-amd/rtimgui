#pragma once

#include <hip/hip_vector_types.h>
// GLAD
#include <glad/glad.h>

#include <include/GLFW/glfw3.h> // Will drag system OpenGL headers
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"


#include <string>
#include "assert.h"
#include <memory>


void glfw_error_callback(int error, const char* description);

struct Window final
{
private:
    struct glfwWindowDestroyer
    {
        void operator()(GLFWwindow* win) const
        {
            glfwDestroyWindow(win);
            glfwTerminate();
        }
    };

public:
    Window(int width, int height, std::string title);

    bool ShouldClose() const;
    void SwapBuffers() const;
    int2 WindowSize() const;
    void PollEvents() const;

    std::unique_ptr<GLFWwindow, glfwWindowDestroyer> nativeWindow;

private:
    
    int width;
    int height;
    std::string title;
};


class MainDisplayWindow
{
public:

    MainDisplayWindow();
    ~MainDisplayWindow();
    bool ShouldClose();
    void PollEvents();
    void Render();
    void Update();
 
private:

    std::unique_ptr<Window> context{nullptr};
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
};