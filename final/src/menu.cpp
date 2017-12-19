#include "menu.hpp"
#include "app.hpp"

using namespace px;

const unsigned char TITLE_FONT_DATA[] = {
#include "font/North_to_South.dat"
};

Menu::Menu(Option *opt)
    : state(State::Pause),
      rectangle_shader(nullptr), text_shader(nullptr), opt(opt), font_size(40),
      on{false},
      fbo(0), bto(0), rbo(0)
{
}

Menu::~Menu()
{
    delete rectangle_shader;
    delete text_shader;
}

void Menu::init()
{
    if (rectangle_shader == nullptr)
        rectangle_shader = new RectangleShader;
    if (text_shader == nullptr)
    {
        text_shader = new TextShader;
        text_shader->setFontHeight(static_cast<std::size_t>(font_size));
        title_font = text_shader->addFont(TITLE_FONT_DATA,
                                           sizeof(TITLE_FONT_DATA));
    }
}

void Menu::setFrameCenter(float x, float y)
{
    frame_center_x = x;
    frame_center_y = y;
    button[0] = frame_center_y - 20;
    button[1] = frame_center_y + 20;
    button[2] = frame_center_y + 60;
    button[3] = frame_center_y + 100;
}

void Menu::click(App *app, int button, int button_state, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT &&  button_state == GLFW_PRESS && action == GLFW_RELEASE)
    {
        if (state == State::Option)
        {
            if (on[0]) app->toggleFullscreen();
            else if (on[1]) opt->setInvertY(!opt->invertY());
            else if (on[2]) state = State::Pause;
        }
        else if (app->scene.state == Scene::State::Win || app->scene.state == Scene::State::Lose)
        {
            if (on[0]) app->restart();
        }
        else
        {
            if (on[0]) app->togglePause();
            else if (on[1]) app->restart();
            else if (on[2]) state = State::Option;
            else if (on[3]) app->close();
        }
    }
}

void Menu::cursor(float cursor_x, float cursor_y)
{
    on[0] = false;
    on[1] = false;
    on[2] = false;
    on[3] = false;

    if (cursor_x > frame_center_x - font_size*12.f &&
        cursor_x < frame_center_x + font_size*12.f)
    {
        if (cursor_y > button[0] && cursor_y < button[0] + font_size)
        {
            on[0] = true;
        }
        else if (cursor_y > button[1] && cursor_y < button[1] + font_size)
        {
            on[1] = true;
        }
        else if (cursor_y > button[2] && cursor_y < button[2] + font_size)
        {
            on[2] = true;
        }
        else if (cursor_y > button[3] && cursor_y < button[3] + font_size)
        {
            on[3] = true;
        }
    }
}
void Menu::render(App *app)
{
    renderScene(app);

    rectangle_shader->render(-1.0f, -1.0f, 2.0f, 2.0f,
                             glm::vec4(0.0f, 0.0f, 0.0f, 0.75f), 0);

    if (state == State::Option)
    {
        text_shader->render("option",
                            frame_center_x, frame_center_y-100, 1.0f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);

        text_shader->render("fullscreen",
                            frame_center_x, button[0], on[0] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
        text_shader->render(std::string("Y axis: ") + (opt->invertY() ? "non-inverted" : "inverted"),
                            frame_center_x, button[1], on[1] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
        text_shader->render("back",
                            frame_center_x, button[2], on[2] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
    }
    else if (app->scene.state == Scene::State::Win || app->scene.state == Scene::State::Lose)
    {
        text_shader->render(app->scene.state == Scene::State::Win ? "win" : "loooooose",
                            frame_center_x, frame_center_y-100, 1.0f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::Center);
        text_shader->render("continue",
                            frame_center_x, button[0], on[0] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::Center);
    }
    else
    {
        text_shader->render("pausss...iiing",
                            frame_center_x, frame_center_y-100, 1.0f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);

        text_shader->render("resume",
                            frame_center_x, button[0], on[0] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
        text_shader->render("restart",
                            frame_center_x, button[1], on[1] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
        text_shader->render("option",
                            frame_center_x, button[2], on[2] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
        text_shader->render("quit",
                            frame_center_x, button[3], on[3] ? 0.6f : 0.4f,
                            glm::vec4(1.0f, 1.0f, 1.0f, 1.0f),
                            Anchor::CenterTop);
    }

}

void Menu::renderScene(App *app)
{
    if (fbo == 0)
    {
        glGenFramebuffers(1, &fbo);
        glGenTextures(1, &bto);
        glGenRenderbuffers(1, &rbo);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glBindTexture(GL_TEXTURE_2D, bto);
    glTexImage2D(GL_TEXTURE_2D,
                 0, GL_RGB, app->scene.cam.width, app->scene.cam.height,
                 0, GL_RGB, GL_UNSIGNED_BYTE, 0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindTexture(GL_TEXTURE_2D, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT8, GL_TEXTURE_2D, bto, 0);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, app->scene.cam.width, app->scene.cam.height);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        app->err("Failed to generate frame buffer");

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    app->scene.render();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}