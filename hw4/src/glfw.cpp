#include "glfw.hpp"

using namespace px;

void errorCallback(int code, const char *description)
{
    // TODO a more friendly error prompt
    throw GLFWError("GLFW Error: " + std::string(description) +
                    " with code " + std::to_string(code));
}

void glfw::init()
{
    if (!glfwInit())
        throw std::runtime_error("Failed to initialize GLFW.");
    glfwSetErrorCallback(errorCallback);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

}

void glfw::terminate()
{
    glfwTerminate();
}