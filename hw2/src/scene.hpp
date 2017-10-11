#ifndef PX_CG_SCENE_HPP
#define PX_CG_SCENE_HPP

#include "object.hpp"

#include "stb_image.h"
#include "stb_image_write.h"

#include <unordered_set>
#include <string>
#include <memory>
#include <limits>

namespace px
{
class Scene;
} // namespace px

class px::Scene
{
public:
    int static constexpr DEFAULT_SCENE_WIDTH = 640;
    int static constexpr DEFAULT_SCENE_HEIGHT = 480;
    Light static const DEFAULT_SCENE_BG; // {0, 0, 0}
    int static constexpr DEFAULT_SAMPLING_RADIUS = 1;
    int static constexpr DEFAULT_RECURSION_DEPTH = 5;
    double static constexpr DEFAULT_HIT_MIN_TOL = 1e-12; // minimum tol to check if a ray hits an object or not.
    double static constexpr DEFAULT_HIT_MAX_TOL = std::numeric_limits<double>::max(); // maximum tol to check if a ray hits an object or not

public:
    std::unordered_set<std::shared_ptr<BaseObject> > objects;
    std::unordered_set<std::shared_ptr<BaseLight> > lights;

    struct Color
    {
        std::uint8_t r;
        std::uint8_t g;
        std::uint8_t b;

        template<typename T>
        std::uint8_t clamp(T const & x)
        {
            return x > 255 ? 255 : x < 0 ? 0 : static_cast<std::uint8_t>(x);
        }
        Color &operator=(Light const &c)
        {
            r = clamp(c.x*255);
            g = clamp(c.y*255);
            b = clamp(c.z*255);
            return *this;
        }
        template<typename  T>
        Color &operator=(Vec3<T> const &c)
        {
            r = clamp(c.x*255);
            g = clamp(c.y*255);
            b = clamp(c.z*255);
            return *this;
        }
    };

    union Pixels
    {
        Color* color;
        std::uint8_t* data;
    } pixels;

    int const &width;
    int const &height;

    Light const &bg;
    Light const &ambient;
    int const &sampling_radius;
    int const &recursion_depth;
    double const &hit_min_tol;
    double const &hit_max_tol;

    std::shared_ptr<Camera> const & cam;

    Scene();
    ~Scene();

    void clearPixels();

    void setSceneSize(int const &width, int const &height);

    void setBackground(double const &light_r,
                       double const &light_g,
                       double const &light_b);
    void setBackground(Light const &light);

    void setCamera(std::shared_ptr<Camera> const &cam);

    void setAmbientLight(Vec3<double> const &light);

    void setSamplingRadius(int const &radius);
    void setRecursionDepth(int const &depth);
    void setHitMinTol(double const &tol);
    void setHitMaxTol(double const &tol);

    void render();

protected:
    Light trace(Ray const &ray,
                double const &refractive_index = 1.0,
                int const &depth = 0);
    Light ambientReflect(Light const &light,
                         Light const &material);
    Light diffuseReflect(Light const &light,
                         Light const &material,
                         Direction const &to_light_vec,
                         Direction const &norm_vec);
    Light specularReflect(Light const &light,
                          Light const &material,
                          Direction const &to_light_vec,
                          Direction const &reflect_vec,
                          int const &specular_exponent);

private:
    int _width;
    int _height;
    Light _bg;
    Light _ambient;
    int _sampling_radius;
    int _recursion_depth;
    double _hit_min_tol;
    double _hit_max_tol;

    std::shared_ptr<Camera> _cam;


};
#endif // PX_CG_SCENE_HPP