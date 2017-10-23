#include "object/geometry/base_geometry.hpp"

using namespace px;

BaseGeometry::BaseGeometry(std::shared_ptr<BaseMaterial> const &material,
                           std::shared_ptr<Transformation> const &trans,
                           int const &n_vertices)
        : material(material),
          transform(trans),
          _raw_vertices(n_vertices, {0, 0, 0})
{}


BaseGeometry *BaseGeometry::hit(Ray const &ray,
                                double const &range_start,
                                double const &range_end,
                                double &hit_at)
{
    // TODO bump mapping
    if (transform == nullptr)
        return hitCheck(ray, range_start, range_end, hit_at);

    Ray trans_ray(transform->point(ray.original), transform->direction(ray.direction));

    return hitCheck(trans_ray, range_start, range_end, hit_at);
}

Direction BaseGeometry::normal(double const &x,
                               double const &y,
                               double const &z)
{
    // TODO bump mapping
    if (transform == nullptr)
        return normalVec(x, y, z);
    return transform->normal(normalVec(transform->point(x, y, z)));
}

Vec3<double> BaseGeometry::textureCoord(double const &x,
                                        double const &y,
                                        double const &z)
{
    if (transform == nullptr)
        return getTextureCoord(x, y, z);
    return getTextureCoord(transform->point(x, y, z));
}


std::shared_ptr<BaseGeometry> Ellipsoid::create(Point const &center_of_bottom_face,
                                               double const &radius_x,
                                               double const &radius_y,
                                                double const &radius_z,
                                               std::shared_ptr<BaseMaterial> const &material,
                                                std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Ellipsoid(center_of_bottom_face,
                                                       radius_x,
                                                       radius_y,
                                                       radius_z,
                                                       material, trans));
}

Ellipsoid::Ellipsoid(Point const &center_of_bottom_face,
                     double const &radius_x, double const &radius_y,
                     double const &radius_z,
                    std::shared_ptr<BaseMaterial> const &material,
                     std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          center(_center),
          radius_x(_radius_x),
          radius_y(_radius_y),
          radius_z(_radius_z)
{
    setParams(center_of_bottom_face,
              radius_x, radius_y, radius_z);
}


BaseGeometry *Ellipsoid::hitCheck(Ray const &ray,
                                  double const &range_start,
                                  double const &range_end,
                                  double &hit_at)
{
    auto xo = ray.original.x - center.x;
    auto yo = ray.original.y - center.y;
    auto zo = ray.original.z - center.z;

    // @see http://www.bmsc.washington.edu/people/merritt/graphics/quadrics.html
    auto A =  _a * ray.direction.x * ray.direction.x +
              _b * ray.direction.y * ray.direction.y +
              _c * ray.direction.z * ray.direction.z;
    auto B =  2 * _a * xo * ray.direction.x +
              2 * _b * yo * ray.direction.y +
              2 * _c * zo * ray.direction.z;
    auto C =  _a * xo * xo +
              _b * yo * yo +
              _c * zo * zo +
              -1;

    if (A == 0)
    {
        if (B == 0) return nullptr;

        auto tmp = - C / B;
        if (tmp > range_start && tmp < range_end)
        {
            hit_at = tmp;
            return this;
        }
        return nullptr;
    }

    auto discriminant = B * B - 4 * A * C;
    if (discriminant < 0)
        return nullptr;

    discriminant = std::sqrt(discriminant);
    auto tmp1 = (-B - discriminant)/ (2.0 * A);
    auto tmp2 = (-B + discriminant)/ (2.0 * A);
    if (tmp1 > tmp2)
        std::swap(tmp1, tmp2);
    if (tmp1 > range_start && tmp1 < range_end)
    {
        hit_at = tmp1;
        return this;
    }
    if (tmp2 > range_start && tmp2 < range_end)
    {
        hit_at = tmp2;
        return this;
    }

    return nullptr;
}

Vec3<double> Ellipsoid::getTextureCoord(double const &x,
                                     double const &y,
                                     double const &z)
{
    auto dx = x - center.x;
    auto dy = y - center.y;
    auto dz = z - center.z;

    return {(1 + std::atan2(dz, dx) / PI) * 0.5,
            std::acos(dy / (dx*dx+dy*dy+dz*dz)) / PI,
            0};;
}

Direction Ellipsoid::normalVec(double const &x, double const &y,
                               double const &z)
{
    return {_a * (x - center.x),
            _b * (y - center.y),
            _c * (z - center.z)};
}

void Ellipsoid::setParams(Point const &center_of_bottom_face,
                          double const &radius_x,
                          double const &radius_y,
                          double const &radius_z)
{
    _center = center_of_bottom_face;
    _radius_x = radius_x;
    _radius_y = radius_y;
    _radius_y = radius_z;

    _a = 1.0 / (radius_x*radius_x);
    _b = 1.0 / (radius_y*radius_y);
    _c = 1.0 / (radius_z*radius_z);

    auto top = center.z + _radius_z;
    auto bottom = center.z - radius_z;

    _raw_vertices.at(4).x = center.x - radius_x;
    _raw_vertices.at(4).y = center.y - radius_y;
    _raw_vertices.at(4).z = top;
    _raw_vertices.at(5).x = center.x - radius_x;
    _raw_vertices.at(5).y = center.y + radius_y;
    _raw_vertices.at(5).z = top;
    _raw_vertices.at(6).x = center.x + radius_x;
    _raw_vertices.at(6).y = center.y + radius_y;
    _raw_vertices.at(6).z = top;
    _raw_vertices.at(7).x = center.x + radius_x;
    _raw_vertices.at(7).y = center.y - radius_y;
    _raw_vertices.at(7).z = top;

    _raw_vertices.at(0).x = center.x - radius_x;
    _raw_vertices.at(0).y = center.y + radius_y;
    _raw_vertices.at(0).z = bottom;
    _raw_vertices.at(1).x = center.x + radius_x;
    _raw_vertices.at(1).y = center.y + radius_y;
    _raw_vertices.at(1).z = bottom;
    _raw_vertices.at(2).x = center.x + radius_x;
    _raw_vertices.at(2).y = center.y - radius_y;
    _raw_vertices.at(2).z = bottom;
    _raw_vertices.at(3).x = center.x + radius_x;
    _raw_vertices.at(3).y = center.y - radius_y;
    _raw_vertices.at(3).z = bottom;
}


std::shared_ptr<BaseGeometry> Box::create(double const &x1, double const &x2,
                                          double const &y1, double const &y2,
                                          double const &z1, double const &z2,
                                          std::shared_ptr<BaseMaterial> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Box(x1, x2,
                                                 y1, y2,
                                                 z1, z2,
                                                 material, trans));
}
std::shared_ptr<BaseGeometry> Box::create(Point const &v1, Point const &v2,
                                          std::shared_ptr<BaseMaterial> const &material,
                                          std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new Box(v1, v2,
                                                 material, trans));
}

Direction const Box::NORM001 = {0, 0, -1};
Direction const Box::NORM002 = {0, 0,  1};
Direction const Box::NORM010 = {0, -1, 0};
Direction const Box::NORM020 = {0,  1, 0};
Direction const Box::NORM100 = {-1, 0, 0};
Direction const Box::NORM200 = { 1, 0, 0};

Box::Box(double const &x1, double const &x2,
         double const &y1, double const &y2,
         double const &z1, double const &z2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 8),
          AbstractBox(x1, y1, z1)
{
    setVertices(x1, x2, y1, y2, z1, z2);
}

Box::Box(Point const &v1, Point const &v2,
         std::shared_ptr<BaseMaterial> const &material,
         std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 0),
          AbstractBox(v1)
{
    setVertices(v1, v2);
}


BaseGeometry *Box::hitCheck(Ray const &ray,
                            double const &t_start,
                            double const &t_end,
                            double &hit_at)
{
    if (AbstractBox::hit(ray, t_start, t_end, hit_at))
        return this;
    else
        return nullptr;
}

Direction Box::normalVec(double const &x, double const &y, double const &z)
{
    if (std::abs(x-_vertex_min.x) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return NORM100;
        }
    }
    else if (std::abs(x-_vertex_max.x) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return NORM200;
        }
    }
    else if (std::abs(y-_vertex_min.y) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return NORM010;
        }
    }
    else if (std::abs(y-_vertex_max.y) < 1e-12)
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return NORM020;
        }
    }
    else if (std::abs(z-_vertex_min.z) < 1e-12)
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return NORM001;
        }
    }
    else if (std::abs(z-_vertex_max.z) < 1e-12)
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return NORM002;
        }
    }
    return {x-_center.x, y-_center.y, z-_center.z}; // Undefined action
}

Vec3<double> Box::getTextureCoord(double const &x,
                               double const &y,
                               double const &z)
{
    if (std::abs(x-_vertex_min.x) < 1e-12) // left side
    {
        if (!(z <= _vertex_min.z || z > _vertex_max.z || y < _vertex_min.y || y > _vertex_max.y))
        {
            return {_vertex_max.z - z, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(x-_vertex_max.x) < 1e-12) // right side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && y > _vertex_min.y && y < _vertex_max.y)
        {
            return {_side.z + _side.x + z - _vertex_min.z, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(y-_vertex_min.y) < 1e-12) // bottom side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _side.z + _side.y + z - _vertex_min.z, 0};
        }
    }
    else if (std::abs(y-_vertex_max.y) < 1e-12) // top side
    {
        if (z > _vertex_min.z && z < _vertex_max.z && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _vertex_max.z - z, 0};
        }
    }
    else if (std::abs(z-_vertex_min.z) < 1e-12) // forward side
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + x - _vertex_min.x, _side.z + _vertex_max.y - y, 0};
        }
    }
    else if (std::abs(z-_vertex_max.z) < 1e-12) // backward side
    {
        if (y > _vertex_min.y && y < _vertex_max.y && x > _vertex_min.x && x < _vertex_max.x)
        {
            return {_side.z + _side.z + _side.z + _vertex_max.x - x, _side.z + _vertex_max.y - y, 0};
        }
    }
    return {x-_center.x, y-_center.y, z-_center.z}; // Undefined action
}

void Box::setVertices(double const &x1, double const &x2,
                      double const &y1, double const &y2,
                      double const &z1, double const &z2)
{
    _vertex_min.x = x1 > x2 ? (_vertex_max.x = x1, x2) : (_vertex_max.x = x2, x1);
    _vertex_min.y = y1 > y2 ? (_vertex_max.y = y1, y2) : (_vertex_max.y = y2, y1);
    _vertex_min.z = z1 > z2 ? (_vertex_max.z = z1, z2) : (_vertex_max.z = z2, z1);

    _center.x = (_vertex_min.x + _vertex_max.x) * 0.5;
    _center.y = (_vertex_min.y + _vertex_max.y) * 0.5;
    _center.z = (_vertex_min.z + _vertex_max.z) * 0.5;

    _side.x = _vertex_max.x - _vertex_min.x;
    _side.y = _vertex_max.y - _vertex_min.y;
    _side.z = _vertex_max.z - _vertex_min.z;

    AbstractBox::_vertices.at(0).x = _vertex_min.x;
    AbstractBox::_vertices.at(0).y = _vertex_min.y;
    AbstractBox::_vertices.at(0).z = _vertex_min.z;

    AbstractBox::_vertices.at(1).x = _vertex_max.x;
    AbstractBox::_vertices.at(1).y = _vertex_min.y;
    AbstractBox::_vertices.at(1).z = _vertex_min.z;

    AbstractBox::_vertices.at(2).x = _vertex_min.x;
    AbstractBox::_vertices.at(2).y = _vertex_max.y;
    AbstractBox::_vertices.at(2).z = _vertex_min.z;

    AbstractBox::_vertices.at(3).x = _vertex_min.x;
    AbstractBox::_vertices.at(3).y = _vertex_min.y;
    AbstractBox::_vertices.at(3).z = _vertex_max.z;

    AbstractBox::_vertices.at(4).x = _vertex_max.x;
    AbstractBox::_vertices.at(4).y = _vertex_max.y;
    AbstractBox::_vertices.at(4).z = _vertex_min.z;

    AbstractBox::_vertices.at(5).x = _vertex_max.x;
    AbstractBox::_vertices.at(5).y = _vertex_min.y;
    AbstractBox::_vertices.at(5).z = _vertex_max.z;

    AbstractBox::_vertices.at(6).x = _vertex_min.x;
    AbstractBox::_vertices.at(6).y = _vertex_max.y;
    AbstractBox::_vertices.at(6).z = _vertex_max.z;

    AbstractBox::_vertices.at(7).x = _vertex_max.x;
    AbstractBox::_vertices.at(7).y = _vertex_max.y;
    AbstractBox::_vertices.at(7).z = _vertex_max.z;
}

std::vector<Point> const & Box::rawVertices() const noexcept
{
    return AbstractBox::_vertices;
}

BoundBox::BoundBox(std::shared_ptr<Transformation> const &trans)
    : BaseGeometry(nullptr, trans, 0), AbstractBox()
{}


BaseGeometry *BoundBox::hitCheck(Ray const &ray,
                                 double const &range_start,
                                 double const &range_end,
                                 double &hit_at)
{
    double t;
    if (AbstractBox::hit(ray, range_start, range_end, t))
    {
        BaseGeometry *obj = nullptr, *tmp;

        double end_range = range_end;
        for (const auto &o : _objs)
        {
            tmp = o->hit(ray, range_start, end_range, t);

            if (tmp == nullptr)
                continue;

            end_range = t;
            obj = tmp;
        }
        if (obj != nullptr)
        {
            hit_at = end_range;
            return obj;
        }
    }
    return nullptr;
}

void BoundBox::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _objs.insert(obj);
    addVertex(obj->rawVertices());
}

std::vector<Point> const & BoundBox::rawVertices() const noexcept
{
    return AbstractBox::_vertices;
}

Direction BoundBox::normalVec(double const &x, double const &y, double const &z)
{
    return {};
}

Vec3<double> BoundBox::getTextureCoord(double const &x,
                                        double const &y,
                                        double const &z)
{
    return {};
}

Direction const BVH::NORM_VEC[7] = {
        { 1,  0, 0},
        { 0,  1, 0},
        { 0,  0, 1},
        { 1,  1, 1},
        {-1,  1, 1},
        {-1, -1, 1},
        { 1, -1, 1}
};

BVH::Extent::Extent(std::shared_ptr<BaseGeometry> const &obj)
    : obj(obj)
{
    for (auto i = 0; i < 7; ++i)
    {
        for (const auto &v: obj->rawVertices())
        {
            auto b = BVH::NORM_VEC[i].dot(v);
            if (b < lower_bound[i]) lower_bound[i] = b;
            if (b > upper_bound[i]) upper_bound[i] = b;
        }
    }
}

BVH::BVH()
    : BaseGeometry(nullptr, nullptr, 0)
{}


BaseGeometry *BVH::hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at)
{
//    double num[7];
//    double den[7];
//    for (auto i = 0; i < 7; ++i) {
//        num[i] = NORM_VEC[i].dot(ray.original);
//        den[i] = NORM_VEC[i].dot(ray.direction);
//    }

    BaseGeometry *obj = nullptr;
    double t;
    double end_range = range_end;
    for (const auto &ex : _extents)
    {
//        bool hit= true;
//        auto tn = -std::numeric_limits<double>::max();
//        auto tf =  std::numeric_limits<double>::max();
//        for (auto i = 0; i < 7; ++i)
//        {
//            auto tmp_tn = (ex.lower_bound[i] - num[i]) / den[i];
//            auto tmp_tf = (ex.upper_bound[i] - num[i]) / den[i];
//            if (den[i] < 0)
//                std::swap(tmp_tn, tmp_tf);
//
//            if (tmp_tn > tn)
//                tn = tmp_tn;
//            if (tmp_tf < tf)
//                tf = tmp_tf;
//
//            if (tn > tf)
//            {
//                hit = false;
//                break;
//            }
//        }
//
//        if (hitCheck && tn < end_range)
//        {
            auto tmp = ex.obj->hit(ray, range_start, end_range, t);
            if (tmp != nullptr)
            {
                end_range = t;
                obj = tmp;
            }
//        }
    }

    if (obj != nullptr)
    {
        hit_at = end_range;
        return obj;
    }

    return nullptr;
}

void BVH::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _extents.push_back(obj);
}

Direction BVH::normalVec(double const &x, double const &y, double const &z)
{
    return {};
}

Vec3<double> BVH::getTextureCoord(double const &x,
                                    double const &y,
                                    double const &z)
{
    return {};
}


std::shared_ptr<BaseGeometry> NormalTriangle::create(Point const &vertex1, Direction const &normal1,
                                                   Point const &vertex2, Direction const &normal2,
                                                   Point const &vertex3, Direction const &normal3,
                                                   std::shared_ptr<BaseMaterial> const &material,
                                                     std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<BaseGeometry>(new NormalTriangle(vertex1, normal1,
                                                          vertex2, normal2,
                                                          vertex3, normal3,
                                                          material, trans));
}

NormalTriangle::NormalTriangle(Point const &vertex1, Direction const &normal1,
                               Point const &vertex2, Direction const &normal2,
                               Point const &vertex3, Direction const &normal3,
                               std::shared_ptr<BaseMaterial> const &material,
                               std::shared_ptr<Transformation> const &trans)
        : BaseGeometry(material, trans, 3)
{}


BaseGeometry *NormalTriangle::hitCheck(Ray const &ray,
                                       double const &t_start,
                                       double const &t_end,
                                       double &hit_at)
{
    return nullptr;
}

Direction NormalTriangle::normalVec(double const &x, double const &y,
                                    double const &z)
{
    return {};
}

Vec3<double> NormalTriangle::getTextureCoord(double const &x, double const &y,
                                          double const &z)
{
    return {};
}
