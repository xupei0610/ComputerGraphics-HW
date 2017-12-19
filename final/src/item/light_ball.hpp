#ifndef PX_CG_ITEM_LIGHT_BALL_HPP
#define PX_CG_ITEM_LIGHT_BALL_HPP

#include "item.hpp"

namespace px { namespace item
{
class LightBall;
class LightBallDroppable;
} }

class px::item::LightBall : public Item
{
private:
    static ItemInfo ITEM_INFO;
    const static float VERTICES[];
public:
    static Shader *shader;
    static unsigned int vao, vbo;
    static void initShader();

    glm::vec3 move_radius;
    float move_speed;
    Light light_source;

    static std::shared_ptr<Item> create();

    void place(glm::vec3 const &p) override;
    glm::vec3 pos() override;

    glm::vec3 halfSize() override;
    void setHalfSize(glm::vec3 const &r) override;

    bool lighting() override;
    const Light & light() override;

    bool preRender() override;
    bool postRender() override;
    void render(glm::mat4 const &view, glm::mat4 const &proj) override;

    void init(Shader *scene_shader) override;
    void update(float dt) override;
    bool canMove() override;

    LightBall(glm::vec3 const &pos = glm::vec3(0.f, 0.f, 0.f),
              glm::vec3 const &scale = glm::vec3(1.f, 1.f, 1.f),
              glm::vec3 const &move_radius = glm::vec3(0.f, 0.f, 0.f),
              float move_speed = 0.f);
    ~LightBall();

    static std::size_t regItem();
protected:
    glm::vec3 org_position;
    glm::vec3 scale;

    glm::vec3 tar;
    glm::vec3 dm;
};

class px::item::LightBallDroppable : public Item
{
private:
    static ItemInfo ITEM_INFO;
public:
    glm::vec3 velocity;
    float m;
    bool is_moving;
    float time_elapsed;
    Light light_source;

public:
    static std::shared_ptr<Item> create();

    static std::size_t regItem();
    void place(glm::vec3 const &p) override;
    glm::vec3 pos() override;

    glm::vec3 halfSize() override;
    void setHalfSize(glm::vec3 const &r) override;

    bool lighting() override;
    const Light & light() override;

    bool preRender() override;
    bool postRender() override;
    void render(glm::mat4 const &view, glm::mat4 const &proj) override;

    void init(Shader *scene_shader) override;
    void update(float dt) override;
    bool canMove() override;
    void hit(glm::vec3 const &) override;

    bool isRigidBody() override;

    float mass() override;
    LightBallDroppable(glm::vec3 const &start_pos = glm::vec3(0.f, 0.f, 0.f),
                       glm::vec3 const &velocity = glm::vec3(1.f, 0.f, 1.f),
                       glm::vec3 const &scale = glm::vec3(1.f),
                       float mass = 0.2f);

    ~LightBallDroppable();
protected:
    glm::vec3 scale;

};

#endif
