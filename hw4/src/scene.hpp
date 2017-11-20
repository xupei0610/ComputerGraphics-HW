#ifndef PX_CG_SCENE_HPP
#define PX_CG_SCENE_HPP

#include <string>
#include "maze.hpp"
#include "character.hpp"
#include "shader/base_shader.hpp"

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

public:
    Scene(Option &opt);

    void setState(State s);
    void init();
    template<typename ...ARGS>
    void reset(ARGS &&...args);
    void render();
    inline const State &gameState() const noexcept { return state; }
    bool run(float dt);

    [[noreturn]]
    void err(std::string const & msg);

    ~Scene();
    Scene &operator=(Scene const &) = default;
    Scene &operator=(Scene &&) = default;

protected:
    State state;
    Shader *shader;

    unsigned int texture[8], vao[2], vbo[2];
    bool need_update_vbo_data;
    bool moveWithCollisionCheck(glm::vec3 span);
};

#endif
