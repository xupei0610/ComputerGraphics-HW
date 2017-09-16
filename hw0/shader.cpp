#include "shader.hpp"

#include <stdexcept>
#include <exception>

template<typename T>
BasicShader<T>::BasicShader()
    : vertices(nullptr),
      num_vertices(0),
      program(0)
{}

template<typename T>
BasicShader<T>::~BasicShader()
{
    clean();
}

template<typename T>
void BasicShader<T>::compileHelper(const char* const &vertex_src,
                                   const char* const &fragment_src)
{
    if (program != 0)
        glDeleteProgram(program);

    program = glCreateProgram();

    GLuint vert_shader, frag_shader;
    SHADER_COMPILE_HELPER(vert_shader, VERTEX,   program, vertex_src,   clean);
    SHADER_COMPILE_HELPER(frag_shader, FRAGMENT, program, fragment_src, clean);

    glLinkProgram(program);
    OPENGL_ERROR_CHECK(program, Program, LINK, clean);

    glDetachShader(program, vert_shader);
    glDetachShader(program, frag_shader);

    glDeleteShader(vert_shader);
    glDeleteShader(frag_shader);

    glBindFragDataLocation(program, 0, "out_color");

    glGenVertexArrays(1, &vao);
}

template<typename T>
void BasicShader<T>::activate()
{
    glUseProgram(program);
    glBindVertexArray(vao);
}

template<typename T>
void BasicShader<T>::draw(GLenum const &mode)
{
    glDrawArrays(mode, 0, num_vertices);
}

template<typename T>
const GLuint *BasicShader<T>::getProgramId() const noexcept
{
    return &program;
}

template<typename T>
void BasicShader<T>::setVertices(const T * const &vertices, const std::size_t &num_vertices)
{
    this->num_vertices = num_vertices;
    this->vertices = vertices;
}

template<typename T>
void BasicShader<T>::setDepth(const T * const &depth)
{
    this->depth = depth;
}

template<typename T>
void BasicShader<T>::setTimeGap(const float* const &delta_time)
{
    this->delta_time = delta_time;
}

template<typename T>
[[ noreturn ]]
void BasicShader<T>::err(std::string const &err_msg)
{
    throw std::runtime_error(err_msg);
}

template<typename T>
void BasicShader<T>::clean()
{
    glDeleteProgram(program);
    glDeleteVertexArrays(1, &vao);
    vao = 0;
    program = 0;
}

const char * AutoChangeableShader::VERTEX_SHADER =
        "#version 330 core\n"
        "in vec2 pos;"
        "uniform float depth;"
        "uniform vec3 color;"
        "out vec3 frag_color;"
        "void main()"
        "{"
        "  if (gl_VertexID == 0)"
        "    frag_color = vec3(color[0], 0, 0);"
        "  else if (gl_VertexID == 1)"
        "    frag_color = vec3(0, color[1], 0);"
        "  else if (gl_VertexID == 2)"
        "    frag_color = vec3(0, 0, color[2]);"
        "  else"
        "    frag_color = vec3(0, 0, 0);"
        "  gl_Position = vec4(pos, depth, 1.0);"
        "}";

const char * AutoChangeableShader::FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 frag_color;"
        "out vec4 out_color;"
        "void main()"
        "{"
        "  out_color = vec4(frag_color, 1.0);"
        "}";

BasicShader<float> * AutoChangeableShader::create()
{
    return static_cast<BasicShader<float> *>(new AutoChangeableShader);
}

AutoChangeableShader::AutoChangeableShader()
    : BasicShader<float>(),
      vbo(0),
      z(0),
      c(0),
      des(new bool[3]),
      color(new float[3])
{
    std::fill_n(des, 3, true);
    std::fill_n(des, 3, 1.0f);
}

void AutoChangeableShader::compile()
{
    BasicShader<float>::compileHelper(VERTEX_SHADER, FRAGMENT_SHADER);

    auto pos = glGetAttribLocation(program, "pos");
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, 0);

    z = glGetUniformLocation(program, "depth");
    c = glGetUniformLocation(program, "color");
}

void AutoChangeableShader::clean()
{
    BasicShader<float>::clean();
    glDeleteBuffers(1, &vbo);
    delete [] des;
    delete [] color;
    des = nullptr;
    color = nullptr;
}

void AutoChangeableShader::updateVertices()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*2*num_vertices,
                 vertices,
                 GL_DYNAMIC_DRAW);
}

void AutoChangeableShader::updateDepth()
{
    glUniform1f(z, *depth);
}

void AutoChangeableShader::updateColor()
{
    for (auto i = 0; i < 3; ++i)
    {
        if (des[i])
        {
            color[i] -= *delta_time * 0.0001 * (i+1);
            if (color[i] < 0.5)
            {
                des[i] = false;
                color[i] = 0.5;
            }
        }
        else
        {
            color[i] += *delta_time * 0.0001 * (3-i);
            if (color[i] > 1)
            {
                des[i] = true;
                color[i] = 1;
            }
        }
    }
    glUniform3f(c, color[0], color[1], color[2]);
}

const char * LocationBasedShader::VERTEX_SHADER =
        "#version 330 core\n"
        "in vec2 pos;"
        "uniform float depth;"
        "out vec3 frag_color;"
        "float c = ((1+pos[0])/2 + (1+pos[1])/2)/2+0.3;"
        "void main()"
        "{"
        "  if (gl_VertexID == 0)"
        "    frag_color = vec3(c, pos[0], pos[1]);"
        "  else if (gl_VertexID == 1)"
        "    frag_color = vec3(pos[0], c, pos[1]);"
        "  else if (gl_VertexID == 2)"
        "    frag_color = vec3(pos[0], pos[1], c);"
        "  else"
        "    frag_color = vec3(0, 0, 0);"
        "  gl_Position = vec4(pos, depth, 1.0);"
        "}";

const char * LocationBasedShader::FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec3 frag_color;"
        "out vec4 out_color;"
        "void main()"
        "{"
        "  out_color = vec4(frag_color, 1.0);"
        "}";

BasicShader<float>* LocationBasedShader::create()
{
    return static_cast<BasicShader<float> *>(new LocationBasedShader);
}

LocationBasedShader::LocationBasedShader()
    : BasicShader<float>(),
      vbo(0),
      z(0)
{}

void LocationBasedShader::compile()
{
    BasicShader<float>::compileHelper(VERTEX_SHADER, FRAGMENT_SHADER);

    auto pos = glGetAttribLocation(program, "pos");
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, 0);

    z = glGetUniformLocation(program, "depth");
}

void LocationBasedShader::clean()
{
    BasicShader<float>::clean();
    glDeleteBuffers(1, &vbo);
}

void LocationBasedShader::updateVertices()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*2*num_vertices,
                 vertices,
                 GL_DYNAMIC_DRAW);
}

void LocationBasedShader::updateDepth()
{
    glUniform1f(z, *depth);
}

void LocationBasedShader::updateColor()
{}

const char * UniformShader::VERTEX_SHADER =
        "#version 330 core\n"
        "in vec2 pos;"
        "uniform float depth;"
        "void main()"
        "{"
        "   gl_Position = vec4(pos, depth, 1.0);"
        "}";

const char * UniformShader::FRAGMENT_SHADER =
        "#version 330 core\n"
        "uniform vec4 color;"
        "out vec4 out_color;"
        "void main()"
        "{"
        "  out_color = vec4(color);"
        "}";

UniformShader::UniformShader()
    : BasicShader<float>(),
      vbo(0),
      z(0),
      c(0),
      des(true),
      opaque(1)
{}

void UniformShader::compile()
{
    BasicShader<float>::compileHelper(VERTEX_SHADER, FRAGMENT_SHADER);

    auto pos = glGetAttribLocation(program, "pos");
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, 0);

    z = glGetUniformLocation(program, "depth");
    c = glGetUniformLocation(program, "color");

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void UniformShader::clean()
{
    BasicShader<float>::clean();
    glDeleteBuffers(1, &vbo);
}

void UniformShader::updateVertices()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*2*num_vertices,
                 vertices,
                 GL_DYNAMIC_DRAW);
}

void UniformShader::updateDepth()
{
    glUniform1f(z, *depth);
}

void UniformShader::updateColor()
{
    if (des)
    {
        opaque -= *delta_time * 0.0002;
        if (opaque < 0.3)
        {
            opaque = 0.3;
            des = false;
        }
    }
    else
    {
        opaque += *delta_time * 0.0002;
        if (opaque > 1)
        {
            opaque = 1;
            des = true;
        }
    }
    glUniform4f(c, 1, 0, 0, opaque);
}

BasicShader<float> * UniformShader::create()
{
    return static_cast<BasicShader *>(new UniformShader);
}

const std::size_t TextureShader::WIDTH = 512;
const std::size_t TextureShader::HEIGHT = 512;

const unsigned char TextureShader::TEXTURE_DATA[] =
{
    #include "texture.dat"
};

const float TextureShader::UV_POS[] =
{
    1, 1,
    1, 0,
    0, 1,
    0, 0
};

const char * TextureShader::VERTEX_SHADER =
        "#version 330 core\n"
        "in vec2 pos;"
        "in vec2 uv_pos;"
        "uniform float depth;"
        "out vec2 uv;"
        "void main()"
        "{"
        "   gl_Position = vec4(pos, depth, 1.0);"
        "   uv = uv_pos;"
        "}";

const char * TextureShader::FRAGMENT_SHADER =
        "#version 330 core\n"
        "in vec2 uv;"
        "uniform sampler2D texture_sample;"
        "out vec4 out_color;"
        "void main()"
        "{"
        "  out_color = vec4(texture(texture_sample, uv).rgb, 1);"
        "}";

BasicShader<float>* TextureShader::create()
{
    return static_cast<BasicShader *>(new TextureShader);
}

TextureShader::TextureShader()
    : BasicShader<float>(),
      vbo(0),
      tbo(0),
      z(0)
{}

void TextureShader::compile()
{
    BasicShader<float>::compileHelper(VERTEX_SHADER, FRAGMENT_SHADER);

    auto pos = glGetAttribLocation(program, "pos");
    auto uv_pos = glGetAttribLocation(program, "uv_pos");

    glGenBuffers(1, &vbo);
    glGenBuffers(1, &tbo);
    glGenTextures(1, &texture);

    glBindVertexArray(vao);

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glEnableVertexAttribArray(pos);
    glVertexAttribPointer(pos, 2, GL_FLOAT, GL_FALSE, 0, 0);

    glBindBuffer(GL_ARRAY_BUFFER, tbo);
    glEnableVertexAttribArray(uv_pos);
    glVertexAttribPointer(uv_pos, 2, GL_FLOAT, GL_FALSE, 0, 0);
    glBufferData(GL_ARRAY_BUFFER, sizeof(UV_POS), UV_POS, GL_STATIC_DRAW);

    glEnable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, WIDTH, HEIGHT, 0, GL_BGR, GL_UNSIGNED_BYTE, TEXTURE_DATA);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glGenerateMipmap(GL_TEXTURE_2D);

    z = glGetUniformLocation(program, "depth");
    t = glGetUniformLocation(program, "texture_sample");
}

void TextureShader::clean()
{
    BasicShader<float>::clean();
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &tbo);
    glDeleteTextures(1, &texture);
}

void TextureShader::updateVertices()
{
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 sizeof(float)*2*num_vertices,
                 vertices,
                 GL_DYNAMIC_DRAW);
}

void TextureShader::updateDepth()
{
    glUniform1f(z, *depth);
}

void TextureShader::updateColor()
{}
