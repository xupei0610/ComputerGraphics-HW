#ifndef PX_CG_SCENE_HPP
#define PX_CG_SCENE_HPP

#include <list>
#include <string>
#include "maze.hpp"
#include "item.hpp"
#include "character.hpp"
#include "shader/base_shader.hpp"
#include "shader/skybox.hpp"

namespace px {

class Scene;
}


class px::Scene
{
public:
    enum State {
        Win,
        Lose,
        Running,
        Over
    };

    Option &opt;
    Maze maze;
    Character character;

    static const float DEFAULT_DISPLACE_AMP;
    static const float DEFAULT_DISPLACE_MID;
    static const char * FS;
    static const char * VS;
    static const char * LAMP_FS;
    static const char * LAMP_VS;
public:
    Scene(Option &opt);

    void setState(State s);
    void init();
    template<typename ...ARGS>
    void reset(ARGS &&...args);
    void render();
    inline const State &gameState() const noexcept { return state; }
    bool run(float dt);

    inline const bool &needUploadData() const noexcept {return need_upload_data; }
    void uploadData();

    [[noreturn]]
    void err(std::string const & msg);

    ~Scene();
    Scene &operator=(Scene const &) = default;
    Scene &operator=(Scene &&) = default;

protected:
    void turnOnHeadLight();
    void turnOffHeadLight();

    static std::vector<unsigned char *> wall_textures;
    static std::vector<std::pair<int, int> > wall_texture_dim;
    static std::vector<unsigned char *> floor_textures;
    static std::vector<std::pair<int, int> > floor_texture_dim;

    State state;
    std::vector<Item*> keys;
    std::vector<Item*> doors;
    std::list<Item*> interact_objs;

    std::vector<float> wall_v;;
    int n_wall_v;

    Shader *shader;
    SkyBox *skybox;

    unsigned int texture[8], vao[3], vbo[3];
    bool need_upload_data;
    bool moveWithCollisionCheck(Character &character, glm::vec3 span);
};

#endif
