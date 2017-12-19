#ifndef PX_CG_MENU_HPP
#define PX_CG_MENU_HPP

#include "option.hpp"
#include "shader/base_shader.hpp"
#include "shader/rectangle.hpp"
#include "shader/text.hpp"

namespace px
{
class App;
class Menu;
}

class px::Menu
{
public:
    enum class State
    {
        Option,
        Pause
    } state;

    int frame_center_x;
    int frame_center_y;

    Menu(Option *opt);

    void init();
    void render(App * app);

    void setFrameCenter(float x, float y);
    void cursor(float cursor_x, float cursor_y);
    void click(App * app, int button, int button_state, int action);
    void renderScene(App *app);

    ~Menu();


    RectangleShader *rectangle_shader;
    TextShader *text_shader;
protected:
    Option *opt;

    int font_size;
    std::size_t title_font;

    bool on[4];
    int button[4];

    unsigned int fbo, bto, rbo;
};

#endif
