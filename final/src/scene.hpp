#ifndef PX_CG_SCENE_HPP
#define PX_CG_SCENE_HPP

#include <string>

#include "character.hpp"
#include "shader/base_shader.hpp"
#include "shader/rectangle.hpp"
#include "shader/skybox.hpp"
#include "item.hpp"
#include "maze.hpp"

namespace px {
class Scene;
}


class px::Scene
{
public:
    Option *opt;
    Camera cam;
    Character character;

    Maze maze;

    enum State {
        Win,
        Lose,
        Running,
        Over
    } state;

    std::vector<std::shared_ptr<Item> > objs;

    int lvl;

    static const char * GEO_FS;
    static const char * GEO_VS;
    static const char * LIGHT_FS;
    static const char * LIGHT_VS;

public:
    Scene(Option *opt);

    void init();
    bool run(float dt);
    void render();

    template<typename ...ARGS>
    void gen(ARGS &&...args);
    void resize();
    void upload();

    [[noreturn]]
    void err(std::string const & msg);

    ~Scene();
    Scene &operator=(Scene const &) = default;
    Scene &operator=(Scene &&) = default;

    Shader *geo_shader;
protected:
    void turnOnHeadLight();
    void turnOffHeadLight();
    glm::vec3 moveWithCollisionCheck(glm::vec3 const &pos, glm::vec3 const &half_size,
                                         glm::vec3 span, Item *player);

    bool winScreen(float dt);

    Shader *light_shader;
    SkyBox *skybox;

    unsigned int texture[8], vao[3], vbo[3], fbo, bto[5], rbo;
    bool need_upload_data;

    std::vector<std::shared_ptr<Item> > keys;
    std::vector<std::shared_ptr<Item> > doors;
};

#endif
