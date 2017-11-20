#include "shader/base_shader.hpp"

#include <glm/gtc/type_ptr.hpp>

using namespace px;

Shader::Shader(const char *vertex_shader, const char *frag_shader)
    : _pid(0)
{
    init(vertex_shader, frag_shader);
}

Shader::~Shader()
{
    glDeleteProgram(_pid);
}

[[noreturn]]
void Shader::err(std::string const &msg)
{
    throw OpenGLError(msg);
}

void Shader::init(const char *vertex_shader, const char *frag_shader)
{
    if (_pid == 0)
        _pid = glCreateProgram();

    unsigned int vs, fs;
    SHADER_COMPILE_HELPER(vs, VERTEX, _pid, vertex_shader)
    SHADER_COMPILE_HELPER(fs, FRAGMENT, _pid, frag_shader)

    glLinkProgram(_pid);
    OPENGL_ERROR_CHECK(_pid, Program, LINK)

    glDetachShader(_pid, vs);
    glDetachShader(_pid, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
}

void Shader::use()
{
    glUseProgram(_pid);
}

void Shader::set(GLint id, glm::mat4 const &val) const
{
    glUniformMatrix4fv(id, 1, GL_FALSE, glm::value_ptr(val));
}
void Shader::set(std::string const &name, glm::mat4 const &val) const
{
    glUniformMatrix4fv(glGetUniformLocation(_pid, name.c_str()),
                       1, GL_FALSE, glm::value_ptr(val));
}
void Shader::set(const char *name, glm::mat4 const &val) const
{
    glUniformMatrix4fv(glGetUniformLocation(_pid, name),
                       1, GL_FALSE, glm::value_ptr(val));
}

void Shader::set(GLint id, bool val) const
{
    glUniform1i(id, val);
}
void Shader::set(std::string const &name, bool val) const
{
    glUniform1i(glGetUniformLocation(_pid, name.c_str()), val);
}
void Shader::set(const char *name, bool val) const
{
    glUniform1i(glGetUniformLocation(_pid, name), val);
}

void Shader::set(GLint id, int val) const
{
    glUniform1i(id, val);
}
void Shader::set(std::string const &name, int val) const
{
    glUniform1i(glGetUniformLocation(_pid, name.c_str()), val);
}
void Shader::set(const char *name, int val) const
{
    glUniform1i(glGetUniformLocation(_pid, name), val);
}

void Shader::set(GLint id, float val) const
{
    glUniform1f(id, val);
}
void Shader::set(std::string const &name, float val) const
{
    glUniform1f(glGetUniformLocation(_pid, name.c_str()), val);
}
void Shader::set(const char *name, float val) const
{
    glUniform1f(glGetUniformLocation(_pid, name), val);
}

void Shader::set(GLint id, glm::vec3 const &val) const
{
    glUniform3fv(id, 1, glm::value_ptr(val));
}
void Shader::set(std::string const &name, glm::vec3 const &val) const
{
    glUniform3fv(glGetUniformLocation(_pid, name.c_str()),
                 1, glm::value_ptr(val));
}
void Shader::set(const char *name, glm::vec3 const &val) const
{
    glUniform3fv(glGetUniformLocation(_pid, name),
                 1, glm::value_ptr(val));
}

void Shader::set(GLint id, glm::vec4 const &val) const
{
    glUniform4fv(id, 1, glm::value_ptr(val));
}
void Shader::set(std::string const &name, glm::vec4 const &val) const
{
    glUniform4fv(glGetUniformLocation(_pid, name.c_str()),
                 1, glm::value_ptr(val));
}
void Shader::set(const char *name, glm::vec4 const &val) const
{
    glUniform4fv(glGetUniformLocation(_pid, name),
                 1, glm::value_ptr(val));
}