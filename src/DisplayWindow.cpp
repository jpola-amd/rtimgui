#include "DisplayWindow.h"


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

Window::Window(int width, int height, std::string title)
    : nativeWindow(glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr)), width(width), height(height), title(std::move(title))
{
}

bool Window::ShouldClose() const
{
    return glfwWindowShouldClose(nativeWindow.get());
}

void Window::SwapBuffers() const
{
    glfwSwapBuffers(nativeWindow.get());
}

int2 Window::WindowSize() const
{
    return {width, height};
}

void Window::PollEvents() const
{
    glfwPollEvents();
}


MainDisplayWindow::MainDisplayWindow()
{
    int result = glfwInit();
    HIP_ASSERT(GLFW_TRUE == result, "Failed to init GLFWWindow");
    glfwSetErrorCallback(glfw_error_callback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);

    context = std::make_unique<Window>(960, 540, "RT Imagination");
    glfwMakeContextCurrent(context->nativeWindow.get());
  

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cerr << "Failed to initialize OpenGL context" << std::endl;
    }

    HIP_ASSERT(context->nativeWindow != nullptr, "Failed to create context");

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void) io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard; // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;  // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(context->nativeWindow.get(), true);
    ImGui_ImplOpenGL3_Init("#version 130");
}

MainDisplayWindow::~MainDisplayWindow()
{
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwTerminate();
}

bool MainDisplayWindow::ShouldClose()
{
    return context->ShouldClose();
}

void MainDisplayWindow::PollEvents()
{
    context->PollEvents();
}

void MainDisplayWindow::Render()
{
    // Rendering
    bool show_demo_window;
    static float f = 0.0f;
    static int counter = 0;
    {
        ImGuiIO& io = ImGui::GetIO(); 

        ImGui::Begin("Hello, world!"); // Create a window called "Hello, world!" and append into it.

        ImGui::Text("This is some useful text.");          // Display some text (you can use a format strings too)
        ImGui::Checkbox("Demo Window", &show_demo_window); // Edit bools storing our window open/close state
       

        ImGui::SliderFloat("float", &f, 0.0f, 1.0f);             // Edit 1 float using a slider from 0.0f to 1.0f
        ImGui::ColorEdit3("clear color", (float*) &clear_color); // Edit 3 floats representing a color

        if (ImGui::Button("Button")) // Buttons return true when clicked (most widgets return true when edited/activated)
            counter++;
        ImGui::SameLine();
        ImGui::Text("counter = %d", counter);

        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::End();
    
    }
    ImGui::Render();
    int display_w, display_h;
    glfwGetFramebufferSize(context->nativeWindow.get(), &display_w, &display_h);
    glViewport(0, 0, display_w, display_h);
    glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    glfwSwapBuffers(context->nativeWindow.get());
}

void MainDisplayWindow::Update()
{
    // Start the Dear ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}
