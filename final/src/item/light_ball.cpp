#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include "item/light_ball.hpp"
#include "util/random.hpp"

using namespace px;

ItemInfo item::LightBall::ITEM_INFO("Light Ball", "floating", 0, false, false, true);
Shader *item::LightBall::shader = nullptr;
unsigned int item::LightBall::vao = 0;
unsigned int item::LightBall::vbo = 0;

ItemInfo item::LightBallDroppable::ITEM_INFO("Light Ball", "droppable", 0, false, false, true);

const float item::LightBall::VERTICES[] =  {
        // back face
        -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
        1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
        1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 0.0f, // bottom-right
        1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 1.0f, 1.0f, // top-right
        -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 0.0f, // bottom-left
        -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, -1.0f, 0.0f, 1.0f, // top-left
        // front face
        -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
        1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 0.0f, // bottom-right
        1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
        1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 1.0f, 1.0f, // top-right
        -1.0f,  1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 1.0f, // top-left
        -1.0f, -1.0f,  1.0f,  0.0f,  0.0f,  1.0f, 0.0f, 0.0f, // bottom-left
        // left face
        -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
        -1.0f,  1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-left
        -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
        -1.0f, -1.0f, -1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-left
        -1.0f, -1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-right
        -1.0f,  1.0f,  1.0f, -1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-right
        // right face
        1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
        1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
        1.0f,  1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 1.0f, // top-right
        1.0f, -1.0f, -1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 1.0f, // bottom-right
        1.0f,  1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 1.0f, 0.0f, // top-left
        1.0f, -1.0f,  1.0f,  1.0f,  0.0f,  0.0f, 0.0f, 0.0f, // bottom-left
        // bottom face
        -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
        1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 1.0f, // top-left
        1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
        1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 1.0f, 0.0f, // bottom-left
        -1.0f, -1.0f,  1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 0.0f, // bottom-right
        -1.0f, -1.0f, -1.0f,  0.0f, -1.0f,  0.0f, 0.0f, 1.0f, // top-right
        // top face
        -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
        1.0f,  1.0f , 1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
        1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 1.0f, // top-right
        1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 1.0f, 0.0f, // bottom-right
        -1.0f,  1.0f, -1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 1.0f, // top-left
        -1.0f,  1.0f,  1.0f,  0.0f,  1.0f,  0.0f, 0.0f, 0.0f  // bottom-left
};

void item::LightBall::initShader()
{
    if (shader == nullptr)
    {
        auto vs =
#include "shader/glsl/lamp_shader.vs"
        ;
        auto fs =

#include "shader/glsl/lamp_shader.fs"
        ;
        shader = new Shader(vs, fs);
        glGenVertexArrays(1, &vao);
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof(VERTICES), VERTICES,
                     GL_STATIC_DRAW);
        glBindVertexArray(vao);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float),
                              (void *) 0);
//        glEnableVertexAttribArray(1);
//        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
//        glEnableVertexAttribArray(2);
//        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }
}

std::shared_ptr<Item> item::LightBall::create()
{
    return std::shared_ptr<Item>(new LightBall());
}

std::size_t item::LightBall::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(item::LightBall::ITEM_INFO, item::LightBall::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

item::LightBall::LightBall(glm::vec3 const &pos,
                           glm::vec3 const &scale,
                           glm::vec3 const &move_radius, float move_speed)
    : Item(regItem()), move_radius(move_radius), move_speed(move_speed),
      org_position(pos), scale(scale)
{
    position = pos;

    light_source.ambient  = glm::vec3(1.f, 1.f, 1.f);
    light_source.diffuse  = glm::vec3(1.f, 1.f, 1.f);
    light_source.specular = glm::vec3(1.f, 1.f, 1.f);
    light_source.coef     = glm::vec3(0.f, 0.f, 2.f);
}

item::LightBall::~LightBall()
{
}

void item::LightBall::place(glm::vec3 const &p)
{
    org_position = p;
    position = p;
}

glm::vec3 item::LightBall::pos()
{
    return position;
}

glm::vec3 item::LightBall::halfSize()
{
    return scale;
}

void item::LightBall::setHalfSize(glm::vec3 const &r)
{
    scale = r;
}

bool item::LightBall::lighting()
{
    return true;
}

const Light &item::LightBall::light()
{
    return light_source;
}

bool item::LightBall::preRender()
{
    return false;
}

bool item::LightBall::postRender()
{
    return true;
}

void item::LightBall::render(glm::mat4 const &view, glm::mat4 const &proj)
{
    shader->use();
    glm::mat4 model = glm::scale(glm::translate(glm::mat4(), position), scale);
    shader->set("diffuse", glm::vec3(1.f, 1.f, 1.f));
    shader->set("model", model);
    shader->set("view", view);
    shader->set("proj", proj);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void item::LightBall::init(Shader *scene_shader)
{
    initShader();
}

void item::LightBall::update(float dt)
{
    if (move_speed == 0)
        return;
    if (move_speed < 0) move_speed = - move_speed;

    movement = dm*dt*move_speed;

    auto stopped = movement.x == 0 && movement.y == 0 && movement.z == 0;

#define REACH(x)    \
    ((dm.x > 0 &&  position.x + movement.x > tar.x) || (dm.x < 0 && position.x + movement.x < tar.x))

    if (REACH(x) || stopped)
    {
        dm.x = rnd();
        tar.x = org_position.x +
                (move_radius.x > 0 ? (dm.x > 0) - (dm.x < 0)
                                   : (dm.x < 0) - (dm.x > 0)) * move_radius.x;
    }
    if (REACH(y) || stopped)
    {
        dm.y = rnd();
        tar.y = org_position.y +
                (move_radius.y > 0 ? (dm.y > 0) - (dm.y < 0)
                                   : (dm.y < 0) - (dm.y > 0)) * move_radius.y;
        if (tar.y < 0)
        std::cout << tar.y << " " << org_position.y << " " << move_radius.y << std::endl;
    }
    if (REACH(z) || stopped)
    {
        dm.z = rnd();
        tar.z = org_position.z +
                (move_radius.z > 0 ? (dm.z > 0) - (dm.z < 0)
                                   : (dm.z < 0) - (dm.z > 0)) * move_radius.z;
    }

#undef REACH

}

bool item::LightBall::canMove()
{
    return true;
}

std::shared_ptr<Item> item::LightBallDroppable::create()
{
    return std::shared_ptr<Item>(new LightBallDroppable());
}

std::size_t item::LightBallDroppable::regItem()
{
    if (ITEM_INFO.id() == 0)
    {
        Item::reg(item::LightBallDroppable::ITEM_INFO, item::LightBall::create);
        if (ITEM_INFO.id() == 0)
            err("Failed to register Item: " + ITEM_INFO.name);
    }
    return ITEM_INFO.id();
}

item::LightBallDroppable::LightBallDroppable(glm::vec3 const &start_pos,
                                                glm::vec3 const &velocity,
                                                glm::vec3 const &scale,
                                             float mass)
        : Item(regItem()), velocity(velocity), m(mass), is_moving(true), time_elapsed(0.f),
          scale(scale)
{
    position = start_pos;
    light_source.ambient  = glm::vec3(.05f, .05f, .05f);
    light_source.diffuse  = glm::vec3(0.f, 0.f, 0.f);
    light_source.specular = glm::vec3(0.f, 0.f, 0.f);
    light_source.coef     = glm::vec3(0.f, 0.f, 5.f);
}

item::LightBallDroppable::~LightBallDroppable()
{
}

void item::LightBallDroppable::place(glm::vec3 const &p)
{
    position = p;
}

glm::vec3 item::LightBallDroppable::pos()
{
    return position;
}

glm::vec3 item::LightBallDroppable::halfSize()
{
    return scale;
}

void item::LightBallDroppable::setHalfSize(glm::vec3 const &r)
{
    scale = r;
}

bool item::LightBallDroppable::lighting()
{
    return true;
}

const Light &item::LightBallDroppable::light()
{
    return light_source;
}

bool item::LightBallDroppable::preRender()
{
    return false;
}

bool item::LightBallDroppable::postRender()
{
    return true;
}

void item::LightBallDroppable::render(glm::mat4 const &view, glm::mat4 const &proj)
{
    LightBall::shader->use();
    glm::mat4 model = glm::scale(glm::translate(glm::mat4(), position), scale);
    LightBall::shader->set("diffuse", glm::vec3(1.f, 1.f, 1.f));
    LightBall::shader->set("model", model);
    LightBall::shader->set("view", view);
    LightBall::shader->set("proj", proj);
    glBindVertexArray(LightBall::vao);
    glDrawArrays(GL_TRIANGLES, 0, 36);
    glBindVertexArray(0);
}

void item::LightBallDroppable::init(Shader *scene_shader)
{
    LightBall::initShader();
}

void item::LightBallDroppable::update(float dt)
{
    if (is_moving)
    {
        auto delta = dt*0.25f;
#define ATTENUATE(x)                                \
        if (velocity.x > 0)                         \
        {                                           \
            velocity.x -= delta;                    \
            if (velocity.x < 0) velocity.x = 0;     \
        }                                           \
        else if (velocity.x < 0)                    \
        {                                           \
            velocity.x += delta;                    \
            if (velocity.x > 0) velocity.x = 0;     \
        }

        ATTENUATE(x)
        ATTENUATE(y)
        ATTENUATE(z)

        if (velocity.x == 0 && velocity.y == 0 && velocity.z == 0)
        {
            movement.x = 0; movement.y = 0; movement.z = 0;
            is_moving = false;
        }
        else
            movement = velocity * dt;
    }
    else
    {
        movement.x = 0;
        movement.y = 0;
        movement.z = 0;
    }
#undef ATTENUATE
}

bool item::LightBallDroppable::canMove()
{
    return is_moving;
}

float item::LightBallDroppable::mass()
{
    return m;
}

void item::LightBallDroppable::hit(glm::vec3 const &)
{
    if (movement.x == 0 || movement.z == 0)
        is_moving = false;
    else
    {
        auto ratio1 = std::abs(velocity.x);
        auto ratio2 = std::abs(velocity.z);
        float ratio = std::max(ratio1, ratio2);
        if (ratio > 1.f)
        {
            velocity.x /= ratio;
            velocity.z /= ratio;
        }
    }
}

bool item::LightBallDroppable::isRigidBody()
{
    return true;
}