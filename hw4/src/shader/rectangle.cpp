#include "shader/rectangle.hpp"
#include "glfw.hpp"
#include <glm/gtc/matrix_transform.hpp>

using namespace px;

const char *RectangleShader::VS = "#version 330 core\n"
        "layout (location = 0) in vec2 v;"
        "layout (location = 1) in vec2 tex_coords_in;"
        ""
        "out vec2 tex_coords;"
        "flat out int use_texture;"
        ""
        "uniform int use_tex;"
        "uniform mat4 proj;"
        ""
        "void main()"
        "{"
        "   gl_Position = proj * vec4(v, 0.0, 1.0);"
        "   tex_coords = tex_coords_in;"
        "   use_texture = use_tex;"
        "}";

const char *RectangleShader::FS = "#version 330 core\n"
        "out vec4 color;"
        ""
        "uniform vec4 rect_color;"
        "uniform sampler2D texture1;"
        "in vec2 tex_coords;"
        ""
        "flat in int use_texture;"
        "const float offset = 1.0/512;"
        "const vec2 offsets[9] = vec2[]("
        "        vec2(-offset, offset),"
        "        vec2(0.0f,    offset),"
        "        vec2(offset,  offset),"
        "        vec2(-offset, 0.0f),"
        "        vec2(0.0f,    0.0f),"
        "        vec2(offset,  0.0f),"
        "        vec2(-offset, -offset),"
        "        vec2(0.0f,    -offset),"
        "        vec2(offset,  -offset)"
        "    );"
        "const float kernel[9] = float[]("
        "    0.0625, 0.125, 0.0625,"
        "    0.125,  0.25,  0.125,"
        "    0.0625, 0.125, 0.0625"
        ");"
        "void main()"
        "{"
        "   if (use_texture == 1)"
        "   {"
        "       vec3 fin_col = vec3(0.0f);"
        "       for(int i = 0; i < 9; i++)"
        "           fin_col += vec3(texture2D(texture1, tex_coords.st + offsets[i]) * kernel[i]);"
        "       color = vec4(mix(fin_col, rect_color.xyz, rect_color.w), 1.0);"
        "   }"
        "   else"
        "   {"
        "       color = rect_color;"
        "   }"
        "}";

RectangleShader::RectangleShader()
    : Shader(VS, FS), vao(0), vbo(0)
{
    glBindFragDataLocation(pid(), 0, "color");

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void *)(2*sizeof(float)));
    glEnableVertexAttribArray(1);
}

RectangleShader::~RectangleShader()
{
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
}

float vertices[] = {
     // x     y       u     v
        0.0f, 0.0f,   0.0f, 1.0f,
        0.0f, 0.0f,   0.0f, 0.0f,
        0.0f, 0.0f,   1.0f, 0.0f,

        0.0f, 0.0f,   0.0f, 1.0f,
        0.0f, 0.0f,   1.0f, 0.0f,
        0.0f, 0.0f,   1.0f, 1.0f
};

void RectangleShader::render(float x, float y, float width, float height,
                             glm::vec4 const &color,
                             unsigned int texture_id)
{

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDisable(GL_DEPTH_TEST);

    use();
    glBindVertexArray(vao);

    if (width > 2.0f || width < -2.0f)
    {
        int w, h;
        glfwGetFramebufferSize(glfwGetCurrentContext(), &w, &h);
        set("proj", glm::ortho(0.0f, static_cast<float>(w),
                               0.0f, static_cast<float>(h)));
    }
    else
        set("proj", glm::mat4());

    set("rect_color", color);

    set("use_tex", texture_id == 0 ? (glBindTexture(GL_TEXTURE_2D, 0), 0)
                                   : (glBindTexture(GL_TEXTURE_2D, texture_id), 1));

    vertices[0] = x;         vertices[1] = y + height;
    vertices[4] = x;         vertices[5] = y;
    vertices[8] = x + width; vertices[9] = y;

    vertices[12] = x;         vertices[13] = y + height;
    vertices[16] = x + width; vertices[17] = y;
    vertices[20] = x + width; vertices[21] = y + height;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_DYNAMIC_DRAW);

    glDrawArrays(GL_TRIANGLES, 0, 6);
}