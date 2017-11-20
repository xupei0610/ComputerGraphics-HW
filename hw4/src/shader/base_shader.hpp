#ifndef PX_CG_BASE_SHADER_HPP
#define PX_CG_BASE_SHADER_HPP

#include <vector>
#include <string>
#include <GL/glew.h>
#include <glm/glm.hpp>

#include "opengl.hpp"

#define __STR_HELPER(X) #X
#define STR(X) __STR_HELPER(X)

#define OPENGL_ERROR_CHECK(target_var, Shader_or_Program, LINK_or_COMPILE)                      \
    {                                                                                           \
        GLint status;                                                                           \
        glGet##Shader_or_Program##iv(target_var, GL_##LINK_or_COMPILE##_STATUS, &status);       \
        if (status == GL_FALSE)                                                                 \
        {                                                                                       \
            glGet##Shader_or_Program##iv(target_var, GL_INFO_LOG_LENGTH, &status);              \
            std::vector<GLchar> err_msg(status);                                                \
            glGet##Shader_or_Program##InfoLog(target_var, status, &status, &err_msg[0]);        \
            err(std::string("Failed to compile " STR(target_var) ": ")                          \
                .append(err_msg.begin(), err_msg.end()));                                       \
        }                                                                                       \
    }

#define SHADER_COMPILE_HELPER(target_var, VERTEX_or_FRAGMENT, program, source_code)             \
    {                                                                                           \
        target_var = glCreateShader(GL_##VERTEX_or_FRAGMENT##_SHADER);                          \
        glShaderSource(target_var, 1, &source_code, 0);                                         \
        glCompileShader(target_var);                                                            \
        OPENGL_ERROR_CHECK(target_var, Shader, COMPILE)                                         \
        glAttachShader(program, target_var);                                                    \
    }

#define TEXTURE_BIND_HELPER(texture_id, loc, repeat_mode, filtering, width, height, data, program_id, shader_var)    \
    glBindTexture(GL_TEXTURE_2D, texture_id);                                                                   \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, repeat_mode);                                             \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, repeat_mode);                                             \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filtering);                                           \
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filtering);                                           \
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);                   \
    glGenerateMipmap(GL_TEXTURE_2D);                                                                            \
    glUniform1i(glGetUniformLocation(program_id, shader_var), loc);

namespace px {
class Shader;
};

class px::Shader
{
public:
    Shader(const char *vertex_shader, const char *frag_shader);
    ~Shader();

    Shader &operator=(Shader const &) = delete;
    Shader &operator=(Shader &&) = delete;

    [[noreturn]]
    void err(std::string const &msg);
    void init(const char *vertex_shader, const char *frag_shader);
    void use();

    inline unsigned int &pid() {return _pid;}

    void set(GLint id, bool val) const;
    void set(std::string const &name, bool val) const;
    void set(const char *name, bool val) const;

    void set(GLint id, int val) const;
    void set(std::string const &name, int val) const;
    void set(const char *name, int val) const;

    void set(GLint id, float val) const;
    void set(std::string const &name, float val) const;
    void set(const char *name, float val) const;

    void set(GLint id, glm::vec3 const &val) const;
    void set(std::string const &name, glm::vec3 const &val) const;
    void set(const char *name, glm::vec3 const &val) const;

    void set(GLint id, glm::vec4 const &val) const;
    void set(std::string const &name, glm::vec4 const &val) const;
    void set(const char *name, glm::vec4 const &val) const;

    void set(GLint id, glm::mat4 const &val) const;
    void set(std::string const &name, glm::mat4 const &val) const;
    void set(const char *name, glm::mat4 const &val) const;

private:
    unsigned int _pid;
};

#endif
