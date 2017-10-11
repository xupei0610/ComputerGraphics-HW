#include "object.hpp"

#include <cmath>
#include <cassert>

using namespace px;

Direction::Direction()
    : Vec3<double>()
{
    Vec3::normalize();
}

Direction::Direction(Vec3<double> const &v)
    : Vec3<double>(v)
{
    Vec3::normalize();
}

Direction::Direction(double const &x, double const &y, double const &z)
    : Vec3<double>(x, y, z)
{
    Vec3::normalize();
}

void Direction::set(double const &x, double const &y, double const &z)
{
    Vec3<double>::x = x;
    Vec3<double>::y = y;
    Vec3<double>::z = z;
    Vec3::normalize();
}

Ray::Ray(Point const &o, Direction const &d)
    : original(o), direction(d)
{}

BaseLight::BaseLight(Light const &light, Point const &pos)
        : light(light), position(pos)
{}

std::shared_ptr<BaseLight> DirectionalLight::create(Light const &light,
                                              Point const &pos)
{
    return std::shared_ptr<BaseLight>(new DirectionalLight(light, pos));
}

DirectionalLight::DirectionalLight(Light const &light, Point const &pos)
    : BaseLight(light, pos)
{}

double DirectionalLight::attenuate(double const &x,
                                   double const &y,
                                   double const &z)
{
    return 1.0;
}

std::shared_ptr<BaseLight> PointLight::create(Light const &light,
                                               Point const &pos)
{
    return std::shared_ptr<BaseLight>(new PointLight(light, pos));
}

PointLight::PointLight(Light const &light, Point const &pos)
    : BaseLight(light, pos)
{}

double PointLight::attenuate(double const &x, double const &y, double const &z)
{
    auto nrm2 = (position.x - x)*(position.x - x) +
                (position.y - y)*(position.y - y) +
                (position.z - z)*(position.z - z);
    if (nrm2 == 0)
        return MAX_LIGHT;
    return 1.0 / nrm2;
}

std::shared_ptr<BaseLight> SpotLight::create(Light const &light,
                                             Point const &pos,
                                             Direction const &direction,
                                             double const &half_angle1,
                                             double const &half_angle2,
                                             double const &falloff)
{
    return std::shared_ptr<BaseLight>(new SpotLight(light, pos, direction,
                                                    half_angle1, half_angle2,
                                                    falloff));
}

SpotLight::SpotLight(Light const &light,
                     Point const &pos,
                     Direction const &direction,
                     double const &half_angle1,
                     double const &half_angle2,
                     double const &falloff)
    : BaseLight(light, pos),
      direction(direction),
      inner_half_angle(_inner_ha),
      outer_half_angle(_outer_ha),
      falloff(falloff)
{
    setAngles(half_angle1, half_angle2);
}

void SpotLight::setAngles(double const &half_angle1, double const &half_angle2)
{
    _inner_ha = half_angle1 < 0 ?
        std::fmod(half_angle1, PI2) + PI2 : std::fmod(half_angle1, PI2);
    _outer_ha = half_angle2 < 0 ?
        std::fmod(half_angle2, PI2) + PI2 : std::fmod(half_angle2, PI2);

    if (_inner_ha_cosine > PI)
        _inner_ha_cosine = PI;
    if (_outer_ha_cosine > PI)
        _outer_ha_cosine = PI;

    if (_outer_ha < _inner_ha)
        std::swap(_outer_ha, _inner_ha);

    _inner_ha_cosine = std::cos(_inner_ha);
    _outer_ha_cosine = std::cos(_outer_ha);
    _multiplier = 1.0 / (_outer_ha_cosine - _inner_ha_cosine);
}
double SpotLight::attenuate(double const &x,
                            double const &y,
                            double const &z)
{
    double dx = x-position.x;
    double dy = y-position.y;
    double dz = z-position.z;
    double nrm2 = dx*dx + dy*dy + dz*dz;
    if (nrm2 == 0)
        return MAX_LIGHT;

    double nrm = std::sqrt(nrm2);

    dx /= nrm;
    dy /= nrm;
    dz /= nrm;

    double cosine = direction.x * dx + direction.y * dy + direction.z * dz;

    if (cosine >= _inner_ha_cosine)
        return 1.0/nrm2;
    if (cosine > _outer_ha_cosine)
        return std::pow(((_outer_ha_cosine-cosine)*_multiplier), falloff)/nrm2;
    return 0;
}

std::shared_ptr<BaseMaterial> UniformMaterial::create(Light const &ambient,
                                           Light const &diffuse,
                                           Light const &specular,
                                           int const &specular_exponent,
                                           Light const &transmissive,
                                           double const &refractive_index)
{
    return std::shared_ptr<BaseMaterial>(new UniformMaterial(ambient,
                                                             diffuse,
                                                             specular,
                                                             specular_exponent,
                                                             transmissive,
                                                             refractive_index));
}

UniformMaterial::UniformMaterial(Light const &ambient,
                   Light const &diffuse,
                   Light const &specular,
                   int const &specular_exponent,
                   Light const &transmissive,
                   double const &refractive_index)
    : BaseMaterial(),
      _ambient(ambient),
      _diffuse(diffuse),
      _specular(specular),
      _specular_exponent(specular_exponent),
      _transmissive(transmissive),
      _refractive_index(refractive_index)
{}

Light UniformMaterial::ambient(double const &, double const &, double const &) const
{
    return _ambient;
}
Light UniformMaterial::diffuse(double const &, double const &, double const &) const
{
    return _diffuse;
}
Light UniformMaterial::specular(double const &, double const &, double const &) const
{
    return _specular;
}
int UniformMaterial::specularExponent() const
{
    return _specular_exponent;
}
Light UniformMaterial::transmissive(double const &x, double const &y, double const &z) const
{
    return _transmissive;
}
double UniformMaterial::refractiveIndex() const
{
    return _refractive_index;
}

std::shared_ptr<Camera> Camera::create(Point const &pos,
                                       Direction const &d,
                                       Direction const &u,
                                       double const &ha)
{
    return std::shared_ptr<Camera>(new Camera(pos, d, u, ha));
}

Camera::Camera(Point const &pos,
               Direction const &d,
               Direction const &u,
               double const &ha)
    : direction(_d), up_vector(_u), right_vector(_r)
{
    setPosition(pos);
    setHalfAngle(ha);
    setDirection(d, u);
}

void Camera::setPosition(Point const &pos)
{
    position = pos;
}

void Camera::setHalfAngle(double const ha)
{
    half_angle = ha;
}

void Camera::setDirection(Direction const &d, Direction const &u)
{
    _d = d;
    _u = u;
    _r = d.cross(u);
}

BaseObject::BaseObject(std::shared_ptr<BaseMaterial> const &material)
    : material(material)
{}

std::shared_ptr<BaseObject> Sphere::create(Point const &pos,
                                           double const &r,
                                           std::shared_ptr<BaseMaterial> const &material)
{
    return std::shared_ptr<BaseObject>(new Sphere(pos, r, material));
}

Sphere::Sphere(Point const &pos,
               double const &r,
               std::shared_ptr<BaseMaterial> const &material)
    : BaseObject(material), position(pos), radius(r)
{}

bool Sphere::hit(Ray const &ray,
                 double const &t_start,
                 double const &t_end,
                 double &hit_at)
{
    auto oc = Vec3<double>(ray.original.x - position.x,
                           ray.original.y - position.y,
                           ray.original.z - position.z);
    auto a = ray.direction.dot(ray.direction);
    auto b = ray.direction.dot(oc);
    auto c = oc.dot(oc) - radius*radius;
    auto discriminant = b*b - a*c;
    if (discriminant > 0)
    {
        auto tmp = -std::sqrt(discriminant)/a;
        auto b_by_a = -b/a;
        tmp += b_by_a;
        if (tmp > t_start && tmp < t_end)
        {
            hit_at = tmp;
            return true;
        }
        else
        {
            tmp = 2*b_by_a-tmp;
            if (tmp > t_start && tmp < t_end)
            {
                hit_at = tmp;
                return true;
            }
        }
    }
    return false;
}

bool Sphere::contain(double const &x, double const &y, double const &z)
{
    if ((x-position.x)*(x-position.x)+(y-position.y)*(y-position.y)+(z-position.z)*(z-position.z) > radius*radius)
        return false;
    else
        return true;
}

Direction Sphere::normVec(double const &x, double const &y, double const &z)
{
    return Direction(x - position.x, y - position.y, z - position.z);
}

Vec3<double> Sphere::relativePos(double const &x, double const &y, double const &z)
{
    return Vec3<double>(x - position.x, y - position.y, z - position.z);
}

std::shared_ptr<BaseObject> Triangle::create(Point const &vertex1,
                                             Point const &vertex2,
                                             Point const &vertex3,
                                             std::shared_ptr<BaseMaterial> const &material)
{
    return std::shared_ptr<BaseObject>(new Triangle(vertex1, vertex2, vertex3, material));
}

Triangle::Triangle(Point const &vertex1,
                   Point const &vertex2,
                   Point const &vertex3,
                   std::shared_ptr<BaseMaterial> const &material)
    : BaseObject(material)
{}

bool Triangle::hit(Ray const &ray,
                   double const &t_start,
                   double const &t_end,
                   double &hit_at)
{
    return false;
}

bool Triangle::contain(double const &x, double const &y, double const &z)
{
    return false;
}

Direction Triangle::normVec(double const &x, double const &y, double const &z)
{
    return {};
}

Vec3<double> Triangle::relativePos(double const &x, double const &y, double const &z)
{
    return {};
}

std::shared_ptr<BaseObject> NormalTriangle::create(Point const &vertex1, Direction const &normal1,
                                                   Point const &vertex2, Direction const &normal2,
                                                   Point const &vertex3, Direction const &normal3,
                                                   std::shared_ptr<BaseMaterial> const &material)
{
    return std::shared_ptr<BaseObject>(new NormalTriangle(vertex1, normal1,
                                                          vertex2, normal2,
                                                          vertex3, normal3,
                                                          material));
}

NormalTriangle::NormalTriangle(Point const &vertex1, Direction const &normal1,
                               Point const &vertex2, Direction const &normal2,
                               Point const &vertex3, Direction const &normal3,
                               std::shared_ptr<BaseMaterial> const &material)
    : BaseObject(material)
{}

bool NormalTriangle::hit(Ray const &ray,
                         double const &t_start,
                         double const &t_end,
                         double &hit_at)
{
    return false;
}

bool NormalTriangle::contain(double const &x, double const &y, double const &z)
{
    return false;
}

Direction NormalTriangle::normVec(double const &x, double const &y, double const &z)
{
    return {};
}

Vec3<double> NormalTriangle::relativePos(double const &x, double const &y, double const &z)
{
    return {};
}
