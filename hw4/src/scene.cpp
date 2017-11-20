#include "scene.hpp"
#include "camera.hpp"
#include "shader/base_shader.hpp"
#include "maze.hpp"

#include "resource.hpp"

#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace px;

const char *VS = "#version 330 core\n"
        "layout (location = 0) in vec3 v;"
        "layout (location = 1) in vec2 v_tex_coord;"
        "layout (location = 2) in vec3 v_norm;"
        "layout (location = 3) in vec3 v_tangent;"
        ""
        "out vec2 tex_coords;" // texture coordinates
        "out vec3 frag_pos;"
        "out vec3 t_cam_pos;"
        "out vec3 t_frag_pos;"
        "out vec3 t_light_pos;"
        "out vec3 t_light_dir;"
        ""
        "uniform mat4 model;"
        "uniform mat4 view;"
        "uniform mat4 proj;"
        "uniform vec3 cam_pos;"
        ""
        "struct SpotLight"
        "{"
        "   vec3 pos;"
        "   vec3 dir;"
        "   float cutoff_outer;"
        "   float cutoff_diff;"
        "   "
        "   vec3 ambient;"
        "   vec3 diffuse;"
        "   vec3 specular;"
        ""
        "   float coef_a0;"
        "   float coef_a1;"
        "   float coef_a2;"
        "};"
        ""
        "uniform SpotLight light;"
        ""
        "struct Material"
        "{"
        "   sampler2D diffuse;"
        "   sampler2D normal;"
        "   sampler2D specular;"
        "   sampler2D displace;"
        "   float shininess;"
        ""
        "};"
        ""
        "uniform Material material;"
        ""
        "flat out int flag;"
        "void main()"
        "{"
        "   tex_coords = v_tex_coord;"
        ""
        "   frag_pos = vec3(model * vec4(v, 1.f));"
        ""
        "   mat4 norm_mat = transpose(inverse(model));"
        "   vec3 T = normalize((norm_mat * vec4(v_tangent, 0.f)).xyz);"
        "   vec3 N = normalize((norm_mat * vec4(v_norm, 0.f)).xyz);"
        "   T = normalize(T - dot(T, N) * N);"
        "   vec3 B = cross(N, T);"
        "   mat3 TBN = transpose(mat3(T, B, N));"
        ""
        "   t_cam_pos   = TBN * cam_pos;"
        "   t_frag_pos  = TBN * frag_pos;"
        "   t_light_pos = TBN * light.pos;" // headlight, light pos is the same with cam pos
        "   t_light_dir = TBN * light.dir;"
        ""
        "   gl_Position = proj * view * model * vec4(v, 1.f);"
        "}";

const char *FS= "#version 330 core\n"
        "out vec4 color;"
        ""
        "in vec2 tex_coords;"
        "in vec3 t_cam_pos;"
        "in vec3 t_frag_pos;"
        "in vec3 t_light_pos;"
        "in vec3 t_light_dir;"
        ""
        "struct SpotLight"
        "{"
        "   vec3 pos;"
        "   vec3 dir;"
        "   float cutoff_outer;"
        "   float cutoff_diff;"
        "   "
        "   vec3 ambient;"
        "   vec3 diffuse;"
        "   vec3 specular;"
        ""
        "   float coef_a0;"
        "   float coef_a1;"
        "   float coef_a2;"
        "};"
        "uniform SpotLight light;"
        ""
        ""
        "struct Material"
        "{"
        "   sampler2D diffuse;"
        "   sampler2D normal;"
        "   sampler2D specular;"
        "   sampler2D displace;"
        "   float shininess;"
        ""
        "};"
        ""
        "uniform Material material;"
        "flat in int flag;"
        ""
        "const vec3 global_ambient = vec3(.05f, .05f, .05f);"
        "void main() {"
        "   if (flag == 1)"
        "    {color = vec4(1, 0, 0,1);return;}"
        ""
        "   vec3 N = texture(material.normal, tex_coords).rgb;"
        "   N = normalize(N * 2.f - 1.f);"
        ""
        "   vec3 obj_color = texture(material.diffuse, tex_coords).rgb;"
        ""
        "   vec3 ambient = light.ambient * obj_color;"                       // ambient
        "   "
        "   vec3 L = normalize(t_light_pos - t_frag_pos);"
        "   vec3 diffuse =  max(dot(L, N), 0.f) * light.diffuse * obj_color;" // diffuse
        ""
        "   vec3 V = normalize(t_cam_pos - t_frag_pos);"
//        "   vec3 R = reflect(-L, N);"
        "   vec3 H = normalize(L + V);" // blinn phong
        "   float spec = pow(max(dot(N, H), 0.f), material.shininess);"
        "   vec3 specular = light.specular * spec * texture(material.specular, tex_coords).rgb;"    // specular
        ""
        "   float cosine = -dot(L, normalize(t_light_dir));"
        "   float intensity = max(min((cosine - light.cutoff_outer)/light.cutoff_diff, 1), 0);"
        "   "
        "   float dist = length(t_light_pos - t_frag_pos);"
        "   float atten = 1.f / (light.coef_a0 + light.coef_a1 * dist + light.coef_a2*dist*dist);"
        ""
        "   ambient  *= atten;"
        "   diffuse  *= intensity * atten;"
        "   specular *= intensity * atten;"
        ""
        "   color = vec4(global_ambient * obj_color + ambient + diffuse + specular, 1.f);"
        "}";

Scene::Scene(Option &opt)
    : opt(opt), character(), shader(nullptr), texture{0,0,0,0,0,0,0,0}, vao{0,0}, vbo{0,0}
{}

Scene::~Scene()
{
    glDeleteVertexArrays(2, vao);
    glDeleteBuffers(2, vbo);
    glDeleteTextures(8, texture);
    delete shader;
}

void Scene::setState(State s)
{
    state = s;
}

void Scene::init()
{
    if (shader == nullptr)
        shader = new Shader(VS, FS);

    if (vao[0] == 0)
        glGenVertexArrays(2, vao);
    if (vbo[0] == 0)
        glGenBuffers(2, vbo);
    if (texture[0] == 0)
        glGenTextures(8, texture);

    shader->use();
    glBindFragDataLocation(shader->pid(), 0, "color");
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(5*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(8*sizeof(float)));
    glEnableVertexAttribArray(3);

    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), 0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(3*sizeof(float)));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(5*sizeof(float)));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 11*sizeof(float), (void *)(8*sizeof(float)));
    glEnableVertexAttribArray(3);

}
float floor_v[] = {
     // coordinates     texture    norm            tangent
     // x    y    z     u    v     x    y    z     x    y    z
        0.f, 0.f, 1.f,  0.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        0.f, 0.f, 0.f,  0.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 0.f,  1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,

        0.f, 0.f, 1.f,  0.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 0.f,  1.f, 0.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
        1.f, 0.f, 1.f,  1.f, 1.f,  0.f, 1.f, 0.f,  1.f, 0.f, 0.f,
};
float cube_v[] = {
        0.f, 0.f, 1.f,  0.f, 0.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,
        0.f, 0.f, 0.f,  1.f, 0.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,
        0.f, 1.f, 0.f,  1.f, 1.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,
        0.f, 0.f, 1.f,  0.f, 0.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,
        0.f, 1.f, 0.f,  1.f, 1.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,
        0.f, 1.f, 1.f,  0.f, 1.f, -1.f, 0.f, 0.f,  0.f, 1.f, 0.f,

        1.f, 0.f, 1.f,  0.f, 0.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,
        1.f, 0.f, 0.f,  1.f, 0.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,
        1.f, 1.f, 0.f,  1.f, 1.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,
        1.f, 0.f, 1.f,  0.f, 0.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,
        1.f, 1.f, 0.f,  1.f, 1.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,
        1.f, 1.f, 1.f,  0.f, 1.f,  1.f, 0.f, 0.f,  0.f,-1.f, 0.f,

        0.f, 1.f, 0.f,  1.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,
        0.f, 0.f, 0.f,  1.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,
        1.f, 0.f, 0.f,  0.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,
        0.f, 1.f, 0.f,  1.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,
        1.f, 0.f, 0.f,  0.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,
        1.f, 1.f, 0.f,  0.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f,-1.f,

        0.f, 1.f, 1.f,  0.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f,
        0.f, 0.f, 1.f,  0.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f,
        1.f, 0.f, 1.f,  1.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f,
        0.f, 1.f, 1.f,  0.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f,
        1.f, 0.f, 1.f,  1.f, 0.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f,
        1.f, 1.f, 1.f,  1.f, 1.f,  0.f, 0.f,-1.f,  0.f, 0.f, 1.f
};
std::vector<float> wall_v;
GLsizei wall_vs = 0;

template<typename ...ARGS>
void Scene::reset(ARGS &&...args)
{
    maze.reset(std::forward<ARGS>(args)...);
    state = State::Running;

    auto h = static_cast<float>(maze.height);
    auto w = static_cast<float>(maze.width);

    auto u = w;
    auto v = h;
    auto ws = (opt.cellSize() + opt.wallThickness()) * (w-1)/2 + opt.wallThickness();
    auto hs = (opt.cellSize() + opt.wallThickness()) * (h-1)/2 + opt.wallThickness();
    floor_v[25] =  u; floor_v[47] =  u; floor_v[58] =  u;
    floor_v[4]  =  v; floor_v[37] =  v; floor_v[59] =  v;
    floor_v[22] = ws; floor_v[44] = ws; floor_v[55] = ws;
    floor_v[2]  = hs; floor_v[35] = hs; floor_v[57] = hs;

    wall_v.clear();
    wall_v.reserve(w*h * 150);
    auto ch = opt.cellHeight();
    auto dl = opt.cellSize()+opt.wallThickness();
    for (auto i = 0; i < h; ++i)
    {
        auto y0 = i/2*dl;
        if (i%2 == 1) y0 += opt.wallThickness();
        auto y1 = y0 + (i%2 == 0 ? opt.wallThickness() : opt.cellSize());

        for (auto j = 0; j < w; ++j)
        {
            if (!maze.isWall(j, i))
                continue;

            auto x0 = j/2*dl;
            if (j%2 == 1) x0 += opt.wallThickness();
            auto x1 = x0 + (j%2 == 0 ? opt.wallThickness() : opt.cellSize());

            auto count = 0;
            if (!maze.isWall(j-1, i))
            {   // render left side
                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++]=-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;
            }
            if (!maze.isWall(j+1, i))
            {   // render right side
                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;  cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y1;  cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f; cube_v[count++] =0.f; cube_v[count++] =0.f; cube_v[count++] = -1.f;
            }
            if (!maze.isWall(j, i-1))
            {   //render up/backward side
                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y0;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = -1.f; cube_v[count++] =-1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;
            }
            if (!maze.isWall(j, i+1))
            {   // render down/forward side
                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x0; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =0.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] =  ch; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =0.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;

                cube_v[count++] = x1; cube_v[count++] = 0.f; cube_v[count++] = y1;   cube_v[count++] =1.f; cube_v[count++] =1.f;
                cube_v[count++] = 0;  cube_v[count++] = 0.f; cube_v[count++] = 1.f;  cube_v[count++] =1.f; cube_v[count++] = 0.f; cube_v[count++] = 0.f;
            }
            wall_v.insert(wall_v.end(), cube_v, cube_v + count);
        }
    }

    wall_vs = static_cast<GLsizei>(wall_v.size())/11;

    auto x = (static_cast<float>(maze.player_x) + 1)*dl - opt.cellSize();
    auto y = (static_cast<float>(maze.player_y) + 1)*dl - opt.cellSize();

    std::vector<int> d;
    d.reserve(4);
    auto to_l = maze.isWall(maze.player_x-1, maze.player_y) ? false : (d.push_back(1), true);
    auto to_r = maze.isWall(maze.player_x+1, maze.player_y) ? false : (d.push_back(2), true);
    auto to_u = maze.isWall(maze.player_x, maze.player_y-1) ? false : (d.push_back(3), true);
    auto to_d = maze.isWall(maze.player_x, maze.player_y+1) ? false : (d.push_back(4), true);
    if (to_l && to_u) d.push_back(5);
    if (to_l && to_d) d.push_back(6);
    if (to_r && to_u) d.push_back(7);
    if (to_r && to_d) d.push_back(8);

    std::random_device rd;
    auto i =  rd() % static_cast<int>(d.size());
    if (d[i] == 1)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -180.f);
    else if (d[i] == 2)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 0.f);
    else if (d[i] == 3)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -90.f);
    else if (d[i] == 4)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 90.f);
    else if (d[i] == 5)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -135.0f);
    else if (d[i] == 6)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 135.0f);
    else if (d[i] == 7)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, -45.0f);
    else //if (d[i] == 8)
        character.reset(x*0.5f, character.characterHeight(), y*0.5f, 45.0f);

    // loading textures
    character.clearBag();


    auto a = new unsigned char[1024*1024*3];
    for (auto i = 0; i < 1024 * 1024 * 3; ++i)
    {
        a[i] = 0;
    }
    i = 0;
//    i = rd() % N_FLOOR_TEXTURES;
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(floor_v), floor_v, GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[0], 0, GL_REPEAT, GL_LINEAR, FLOOR_TEXTURE_DIM[i*2], FLOOR_TEXTURE_DIM[i*2+1], FLOOR_TEXTURES[i*4],   shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[1], 1, GL_REPEAT, GL_LINEAR, FLOOR_TEXTURE_DIM[i*2], FLOOR_TEXTURE_DIM[i*2+1], FLOOR_TEXTURES[i*4+1], shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[2], 2, GL_REPEAT, GL_LINEAR, FLOOR_TEXTURE_DIM[i*2], FLOOR_TEXTURE_DIM[i*2+1], FLOOR_TEXTURES[i*4+2], shader->pid(), "material.specular")
    TEXTURE_BIND_HELPER(texture[3], 3, GL_REPEAT, GL_LINEAR, FLOOR_TEXTURE_DIM[i*2], FLOOR_TEXTURE_DIM[i*2+1], FLOOR_TEXTURES[i*4+3], shader->pid(), "material.displace");

//    i = rd() % N_WALL_TEXTURES;
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*wall_v.size(), wall_v.data(), GL_STATIC_DRAW);
    TEXTURE_BIND_HELPER(texture[4], 0, GL_REPEAT, GL_LINEAR, WALL_TEXTURE_DIM[i*2], WALL_TEXTURE_DIM[i*2+1], WALL_TEXTURES[i*4],   shader->pid(), "material.diffuse");
    TEXTURE_BIND_HELPER(texture[5], 1, GL_REPEAT, GL_LINEAR, WALL_TEXTURE_DIM[i*2], WALL_TEXTURE_DIM[i*2+1], WALL_TEXTURES[i*4+1], shader->pid(), "material.normal");
    TEXTURE_BIND_HELPER(texture[6], 2, GL_REPEAT, GL_LINEAR, WALL_TEXTURE_DIM[i*2], WALL_TEXTURE_DIM[i*2+1], WALL_TEXTURES[i*4+2], shader->pid(), "material.specular");
    TEXTURE_BIND_HELPER(texture[7], 3, GL_REPEAT, GL_LINEAR, WALL_TEXTURE_DIM[i*2], WALL_TEXTURE_DIM[i*2+1], WALL_TEXTURES[i*4+3], shader->pid(), "material.displace");

    need_update_vbo_data = true;

#ifndef NDEBUG
    std::cout << maze.map << std::endl;
#endif
}

template void Scene::reset(Map const &);
template void Scene::reset(Maze const &);
template void Scene::reset(std::size_t const &, std::size_t const &);
template void Scene::reset(int const &, int const &);

void Scene::render()
{
//    glEnable(GL_CULL_FACE);
//    glCullFace(GL_BACK);
//    glFrontFace(GL_CCW);
    glEnable(GL_DEPTH_TEST);
    glClearColor(.2f, .3f, .3f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    shader->use();

    if (need_update_vbo_data)
    {
        need_update_vbo_data = false;

        shader->set("light.cutoff_outer", 0.9537f);
        shader->set("light.cutoff_diff", 0.0226f);
        shader->set("light.ambient", glm::vec3(.5f, .5f, .5f));
        shader->set("light.diffuse", glm::vec3(.1f, .1f, .1f));
        shader->set("light.specular", glm::vec3(.0f, .0f, .0f));
        shader->set("light.coef_a0", 1.f);
        shader->set("light.coef_a1", .09f);
        shader->set("light.coef_a2", .032f);
    }


    shader->set("view", character.cam.viewMat());
    shader->set("proj", character.cam.projMat());
    shader->set("model", glm::mat4());
    shader->set("cam_pos", character.cam.cam_pos);
    shader->set("light.pos", character.cam.cam_pos);
    shader->set("light.dir", character.cam.cam_dir);

    // render floor
    glBindVertexArray(vao[0]);
    shader->set("material.shininess", 32.f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[0]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[1]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[2]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[3]);
    glDrawArrays(GL_TRIANGLES, 0, 6);

    // render walls
    glBindVertexArray(vao[1]);
    shader->set("material.shininess", 32.f);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture[4]);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, texture[5]);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, texture[6]);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, texture[7]);
    glDrawArrays(GL_TRIANGLES, 0, wall_vs);
}

bool Scene::run(float dt)
{
#ifndef NDEBUG
    bool moved = false;
#endif
    if (state == State::Running)
#ifndef NDEBUG
        moved =
#endif
            moveWithCollisionCheck(character.makeAction(dt));

    // TODO game over moive
//    else ...

    character.cam.updateViewMat();

#ifndef NDEBUG
    if (moved)
        std::cout << "\n" << maze.map << std::endl;
    std::cout << "\rLocation: ("
              << character.cam.cam_pos.x << ", "
              << character.cam.cam_pos.y << ", "
              << character.cam.cam_pos.z << "); Look at: ("
              << character.cam.cam_dir.x << ", "
              << character.cam.cam_dir.y << ", "
              << character.cam.cam_dir.z << ")"
              << std::flush;
#endif

    // TODO modify the following line for game over movie
    return state == State::Running ? true : false;
}

bool Scene::moveWithCollisionCheck(glm::vec3 span)
{
    auto moved = false;
    auto dl = opt.cellSize()+opt.wallThickness();
    while (span.x != 0.f || span.z != 0.f)
    {
        if (maze.isEndPoint(maze.player_x, maze.player_y))
        {
            state = State::Win;
            break;
        }

        if (character.collectItem(maze.collect(maze.player_x, maze.player_y), 1))
            maze.clear(maze.player_x, maze.player_y);

        if (character.hasItem(maze.keyFor(maze.player_x, maze.player_y)))
        {
            state = State::Win;
            break;
        }
        if (character.currentHp() < 0)
        {
            state = State::Lose;
            break;
        }

        auto x0 = maze.player_x/2*dl;
        if (maze.player_x%2 == 1) x0 += opt.wallThickness();
        auto x1 = x0 + (maze.player_x%2 == 0 ? opt.wallThickness() : opt.cellSize());

        auto z0 = maze.player_y/2*dl;
        if (maze.player_y%2 == 1) z0 += opt.wallThickness();
        auto z1 = z0 + (maze.player_y%2 == 0 ? opt.wallThickness() : opt.cellSize());

        auto dx = (span.x > 0.f ? (x1 - character.cam.cam_pos.x) : (x0 - character.cam.cam_pos.x))/span.x;
        auto dz = (span.z > 0.f ? (z1 - character.cam.cam_pos.z) : (z0 - character.cam.cam_pos.z))/span.z;

        if (span.x != 0.f)
        {
            if (span.x > 0.f && !maze.canMoveRight())
            {
                character.cam.cam_pos.x = std::min(x1 - character.characterHalfSize(),
                                                   character.cam.cam_pos.x + span.x);
                span.x = 0.f;
            }
            else if (span.x < 0.f && !maze.canMoveLeft())
            {
                character.cam.cam_pos.x = std::max(x0 + character.characterHalfSize(),
                                                   character.cam.cam_pos.x + span.x);
                span.x = 0.f;
            }
        }
        if (span.z != 0.f)
        {
            if (span.z > 0.f && !maze.canMoveDown())
            {
                character.cam.cam_pos.z = std::min(z1 - character.characterHalfSize(),
                                                   character.cam.cam_pos.z + span.z);
                span.z = 0.f;
            }
            else if (span.z < 0.f && !maze.canMoveUp())
            {
                character.cam.cam_pos.z = std::max(z0 + character.characterHalfSize(),
                                                   character.cam.cam_pos.z + span.z);
                span.z = 0.f;
            }
        }

        if ((span.z == 0.f || dx < dz) && span.x != 0.f)
        {
            if (dx >= 1.0)
            {
                character.cam.cam_pos.x += span.x;
                span.x = 0.f;
            }
            else
            {
                if (span.x > 0.f)
                    maze.moveRight();
                else
                    maze.moveLeft();
                moved = true;
                if (span.x > 0.f)
                {
                    span.x += character.cam.cam_pos.x - x1;
                    character.cam.cam_pos.x = x1;
                }
                else if (span.x < 0.f)
                {
                    span.x += character.cam.cam_pos.x - x0;
                    character.cam.cam_pos.x = x0;
                }
            }
        }
        else if (span.z != 0.f)
        {
            if (dz >= 1.0)
            {
                character.cam.cam_pos.z += span.z;
                span.z = 0.f;
            }
            else
            {
                if (span.z > 0.f)
                    maze.moveDown();
                else
                    maze.moveUp();
                moved = true;
                if (span.z > 0.f)
                {
                    span.z += character.cam.cam_pos.z - z1;
                    character.cam.cam_pos.z = z1;
                }
                else if (span.z < 0.f)
                {
                    span.z += character.cam.cam_pos.z - z0;
                    character.cam.cam_pos.z = z0;
                }
            }
        }
    }

    return moved;
}