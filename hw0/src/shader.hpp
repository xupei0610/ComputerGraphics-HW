#ifndef SHADER_HPP
#define SHADER_HPP

#include "glad/glad.h"

#include "global.hpp"
#include <string>
#include <vector>

#define OPENGL_ERROR_CHECK(target_var, Shader_or_Program, LINK_or_COMPILE, callback_fn)         \
    {                                                                                           \
        GLint status;                                                                           \
        glGet##Shader_or_Program##iv(target_var, GL_##LINK_or_COMPILE##_STATUS, &status);       \
        if (status == GL_FALSE)                                                                 \
        {                                                                                       \
            glGet##Shader_or_Program##iv(target_var, GL_INFO_LOG_LENGTH, &status);              \
            std::vector<GLchar> err_msg(status);                                                \
            glGet##Shader_or_Program##InfoLog(target_var, status, &status, &err_msg[0]);        \
            callback_fn();                                                                      \
            err(std::string("Failed to compile " STR(target_var) ". ")                          \
                .append(err_msg.begin(), err_msg.end()));                                       \
        }                                                                                       \
    }

#define SHADER_COMPILE_HELPER(target_var, VERTEX_or_FRAGMENT, program, source_code, callback_fn)\
    {                                                                                           \
        target_var = glCreateShader(GL_##VERTEX_or_FRAGMENT##_SHADER);                          \
        glShaderSource(target_var, 1, &source_code, 0);                                         \
        glCompileShader(target_var);                                                            \
        OPENGL_ERROR_CHECK(target_var, Shader, COMPILE, callback_fn)                            \
        glAttachShader(program, target_var);                                                    \
    }

template<typename T>
class BasicShader
{
public:
    virtual void compile() = 0;
    virtual void activate();
    virtual void draw(const GLenum &mode);
    const GLuint* getProgramId() const noexcept;

    virtual void setVertices(const T* const &vertices, std::size_t const &n);
    virtual void setDepth(const T* const &depth);
    virtual void setTimeGap(const float* const &delta_time);
    virtual void updateVertices() = 0;
    virtual void updateDepth() = 0;
    virtual void updateColor() = 0;
    virtual void clean();

    virtual ~BasicShader();

protected:
    const T *vertices;
    const T *depth;
    std::size_t num_vertices;
    GLuint program;
    GLuint vao;

    const float *delta_time;

    BasicShader();

    void compileHelper(const char* const &vertex_src,
                       const char* const &fragment_src);
    [[ noreturn ]]
    void err(std::string const &err_msg);

public:
    DISABLE_DEFAULT_CONSTRUCTOR(BasicShader)
};

class AutoChangeableShader : public BasicShader<float>
{
public:
    const static char * VERTEX_SHADER;
    const static char * FRAGMENT_SHADER;

    static BasicShader<float> * create();

    AutoChangeableShader();

    void compile() override;
    void clean() override;
    void updateVertices() override;
    void updateDepth() override;
    void updateColor() override;

protected:
    GLuint vbo;
    GLuint z; // depth
    GLuint c; // color

    bool  *des;
    float *color;

public:
    DISABLE_DEFAULT_CONSTRUCTOR(AutoChangeableShader)
};

class LocationBasedShader : public BasicShader<float>
{
public:
    const static char * VERTEX_SHADER;
    const static char * FRAGMENT_SHADER;

    static BasicShader<float> * create();

    LocationBasedShader();

    void compile() override;
    void clean() override;
    void updateVertices() override;
    void updateDepth() override;
    void updateColor() override;

protected:
    GLuint vbo;
    GLuint z; // depth

public:
    DISABLE_DEFAULT_CONSTRUCTOR(LocationBasedShader)
};

class UniformShader : public BasicShader<float>
{
public:
    const static char * VERTEX_SHADER;
    const static char * FRAGMENT_SHADER;

    static BasicShader<float> * create();

    UniformShader();

    void compile() override;
    void clean() override;
    void updateVertices() override;
    void updateDepth() override;
    void updateColor() override;

protected:
    GLuint vbo;
    GLuint z; // depth
    GLuint c; // color

    bool des;
    float opaque;

public:
    DISABLE_DEFAULT_CONSTRUCTOR(UniformShader)
};

class TextureShader : public BasicShader<float>
{
public:
    const static std::size_t WIDTH;
    const static std::size_t HEIGHT;
    const static unsigned char TEXTURE_DATA[];
    const static float UV_POS[];

    const static char * VERTEX_SHADER;
    const static char * FRAGMENT_SHADER;

    static BasicShader<float> * create();

    TextureShader();

    void compile() override;
    void clean() override;
    void updateVertices() override;
    void updateDepth() override;
    void updateColor() override;

protected:
    GLuint vbo;
    GLuint tbo;
    GLuint texture;
    GLuint z; // depth
    GLuint t;

public:
    DISABLE_DEFAULT_CONSTRUCTOR(TextureShader)
};

#endif // SHADER_HPP
