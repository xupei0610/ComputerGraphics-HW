#include "shape.hpp"
#include "app.hpp"

#include <cmath>

// solve `constexpr static` bug of clang
template<typename T>
constexpr std::size_t Triangle<T>::NUM_VERTICES;
template<typename T>
constexpr std::size_t Square<T>::NUM_VERTICES;
template<typename T>
constexpr T Point<T>::CLOSE_THRESHOLD;

template<typename T>
BasicShape<T>::BasicShape(std::size_t const &num_vertices,
                          T const &x, T const &y, T const &size, T const &angle)
    : vertices(vert_data),
      x(x),
      y(y),
      size(size),
      angle(angle),
      num_vertices(num_vertices),
      vert_data(num_vertices*2, 0),
      org_x(x),
      org_y(y),
      org_size(size),
      org_angle(angle)
{
    vert_data.shrink_to_fit();
}

template<typename T>
void BasicShape<T>::rotate(Point<T> const &pos, Point<T> const &ref_pos)
{
    angle += std::atan2(pos.y-y, pos.x-x)
            - std::atan2(ref_pos.y-y, ref_pos.x-x);
    if (angle > PI)
        angle -= 2*PI;
    else if (angle < -PI)
        angle += 2*PI;
}

template<typename T>
void BasicShape<T>::rotate(T const &ang)
{
    angle += ang;
    if (angle > PI)
        angle -= 2*PI;
    else if (angle < -PI)
        angle += 2*PI;
}

template<typename T>
void BasicShape<T>::move(T const &offset_x, T const &offset_y)
{
    x += offset_x;
    y += offset_y;
    if (x < -1)
        x = -1;
    else if (x > 1)
        x = 1;
    if (y < -1)
        y = -1;
    else if (y > 1)
        y = 1;
}

template<typename T>
void BasicShape<T>::scaleTo(T const &new_size)
{
    size = new_size;
    if (size < App::MIN_SHAPE_SIZE)
        size = App::MIN_SHAPE_SIZE;
    else if (size > App::MAX_SHAPE_SIZE)
        size = App::MAX_SHAPE_SIZE;
}

template<typename T>
void BasicShape<T>::scale(T const &scale)
{
    size += scale;
    if (size < App::MIN_SHAPE_SIZE)
        size = App::MIN_SHAPE_SIZE;
    else if (size > App::MAX_SHAPE_SIZE)
        size = App::MAX_SHAPE_SIZE;
}

template<typename T>
void BasicShape<T>::reset()
{
    x = org_x;
    y = org_y;
    size = org_size;
    angle = org_angle;
}

template<typename T>
Square<T>::Square(T const &x, T const &y, T const &size, T const &angle)
    : BasicShape<T>(NUM_VERTICES, x, y, size, angle)
{}

template<typename T>
SHAPE_PROTOTYPE Square<T>::getProperty() const
{
    return SQUARE_SHAPE;
}

template<typename T>
void Square<T>::updateVertices(T const &aspect_ratio)
{
    auto c = std::cos(BasicShape<T>::angle);
    auto s = std::sin(BasicShape<T>::angle);

    BasicShape<T>::vert_data[0] = BasicShape<T>::x
            + c * BasicShape<T>::size - s * BasicShape<T>::size;
    BasicShape<T>::vert_data[1] = BasicShape<T>::y
            + s * BasicShape<T>::size + c * BasicShape<T>::size;

    BasicShape<T>::vert_data[2] = BasicShape<T>::x
            + c * BasicShape<T>::size - s * (-BasicShape<T>::size);
    BasicShape<T>::vert_data[3] = BasicShape<T>::y
            + s * BasicShape<T>::size + c * (-BasicShape<T>::size);

    BasicShape<T>::vert_data[4] = BasicShape<T>::x
            + c * (-BasicShape<T>::size) - s * BasicShape<T>::size;
    BasicShape<T>::vert_data[5] = BasicShape<T>::y
            + s * (-BasicShape<T>::size) + c * BasicShape<T>::size;

    BasicShape<T>::vert_data[6] = BasicShape<T>::x
            + c * (-BasicShape<T>::size) - s * (-BasicShape<T>::size);
    BasicShape<T>::vert_data[7] = BasicShape<T>::y
            + s * (-BasicShape<T>::size) + c * (-BasicShape<T>::size);

    if (aspect_ratio > 1)
    {
        for (std::remove_const<decltype(NUM_VERTICES)>::type i = 0; i < NUM_VERTICES; ++i)
            BasicShape<T>::vert_data[i * 2] /= aspect_ratio;
    }
    else if (aspect_ratio < 1)
    {
        for (std::remove_const<decltype(NUM_VERTICES)>::type i = 0; i < NUM_VERTICES; ++i)
            BasicShape<T>::vert_data[i * 2 + 1] *= aspect_ratio;
    }
}

template<typename T>
BasicShape<T> *
Square<T>::create(const T &x, const T &y, const T &size, const T &angle)
{
    return static_cast<BasicShape<T> *>(new Square<T>(x, y, size, angle));
}

template<typename T>
const T Triangle<T>::SQRT_3 = std::sqrt(3);

template<typename T>
Triangle<T>::Triangle(T const &x, T const &y, T const &size, T const &angle)
    : BasicShape<T>(NUM_VERTICES, x, y, size, angle)
{}

template<typename T>
SHAPE_PROTOTYPE Triangle<T>::getProperty() const
{
    return TRIANGLE_SHAPE;
}

template<typename T>
void Triangle<T>::updateVertices(T const &aspect_ratio)
{
    auto c = std::cos(BasicShape<T>::angle);
    auto s = std::sin(BasicShape<T>::angle);

    auto b = -BasicShape<T>::size / 2;
    auto l = SQRT_3 * b;
    auto r = -l;

    BasicShape<T>::vert_data[0] = BasicShape<T>::x + c * 0 - s * BasicShape<T>::size;
    BasicShape<T>::vert_data[1] = BasicShape<T>::y + s * 0 + c * BasicShape<T>::size;

    BasicShape<T>::vert_data[2] = BasicShape<T>::x + c * r - s * b;
    BasicShape<T>::vert_data[3] = BasicShape<T>::y + s * r + c * b;

    BasicShape<T>::vert_data[4] = BasicShape<T>::x + c * l - s * b;
    BasicShape<T>::vert_data[5] = BasicShape<T>::y + s * l + c * b;

    if (aspect_ratio > 1)
    {
        for (std::remove_const<decltype(NUM_VERTICES)>::type i = 0; i < NUM_VERTICES; ++i)
            BasicShape<T>::vert_data[i*2] /= aspect_ratio;
    }
    else if (aspect_ratio < 1)
    {
        for (std::remove_const<decltype(NUM_VERTICES)>::type i = 0; i < NUM_VERTICES; ++i)
            BasicShape<T>::vert_data[i*2+1] *= aspect_ratio;
    }
}

template<typename T>
BasicShape<T> *
Triangle<T>::create(const T &x, const T &y, const T &size, const T &angle)
{
    return static_cast<BasicShape<T> *>(new Triangle<T>(x, y, size, angle));
}

template<typename T>
Point<T>::Point()
    : x(0),
      y(0)
{}

template<typename T>
Point<T>::Point(T const &x, T const &y)
    : x(x),
      y(y)
{}

template<typename T>
RelativePos Point<T>::relativeTo(const BasicShape<T> * const &shape, T const &aspect_ratio) const
{
    switch (shape->getProperty())
    {
    case SQUARE_SHAPE:
        return relate(*this, *dynamic_cast<const Square<T> *>(shape), aspect_ratio);
    case TRIANGLE_SHAPE:
        return relate(*this, *dynamic_cast<const Triangle<T> *>(shape), aspect_ratio);
    default:
        return RelativePos::Outer;
    }
}

template<typename T>
std::ostream &operator <<(std::ostream &os, Point<T> const &p)
{
    return os << p.x << ", " << p.y;
}

template<typename T>
RelativePos relate(Point<T> const &point, Square<T> const &square, T const &aspect_ratio)
{
    T px, py;

    if (aspect_ratio == 1)
    {
        px = point.x;
        py = point.y;
    }
    else if (aspect_ratio > 1)
    {
        px = point.x * aspect_ratio;
        py = point.y;
    }
    else
    {
        px = point.x;
        py = point.y / aspect_ratio;
    }

    auto c = std::cos(square.angle);
    auto s = std::sin(square.angle);

    auto org_x = std::abs( c * (px-square.x) + s * (py-square.y));
    auto org_y = std::abs(-s * (px-square.x) + c * (py-square.y));

    auto gap_x = std::abs(org_x - square.size);
    auto gap_y = std::abs(org_y - square.size);

    if (gap_x < point.CLOSE_THRESHOLD && gap_y < point.CLOSE_THRESHOLD)
        return RelativePos::Corner;
    else if ((gap_x < point.CLOSE_THRESHOLD && org_y < square.size) ||
             (gap_y < point.CLOSE_THRESHOLD && org_x < square.size))
        return RelativePos::Border;
    else if (org_x < square.size && org_y < square.size)
        return RelativePos::Inner;
    return RelativePos::Outer;
}

template<typename T>
RelativePos relate(Point<T> const &point, Triangle<T> const &triangle, T const &aspect_ratio)
{
    T px, py;

    if (aspect_ratio == 1)
    {
        px = point.x;
        py = point.y;
    }
    else if (aspect_ratio > 1)
    {
        px = point.x * aspect_ratio;
        py = point.y;
    }
    else
    {
        px = point.x;
        py = point.y / aspect_ratio;
    }

    auto c = std::cos(triangle.angle);
    auto s = std::sin(triangle.angle);

    auto org_x =  c * (px-triangle.x) + s * (py-triangle.y);
    auto org_y = -s * (px-triangle.x) + c * (py-triangle.y);

    auto b = -triangle.size / 2;
    auto l = triangle.SQRT_3 * b;

    c = -0.5; // cos(2*PI/3)
    s = triangle.SQRT_3 / 2; // sin(2*PI/3)

    bool above = true;

    auto i = 0;
    while (true)
    {
        if (std::abs(org_y - b) < point.CLOSE_THRESHOLD)
        {
            if (std::abs(org_x - l) < point.CLOSE_THRESHOLD)
                return RelativePos::Corner;
            else if (org_x > l && org_x < -l)
                return RelativePos::Border;
        }
        else if (above)
        {
            if (org_y < b)
                above = false;
        }

        if (++i == 3)
            break;

        auto new_org_x = c * org_x - s * org_y;
        auto new_org_y = s * org_x + c * org_y;

        org_x = new_org_x;
        org_y = new_org_y;
    }

    if (above)
        return RelativePos::Inner;

    return RelativePos::Outer;
}


CLASS_TEMPLATE_SPECIFICATION_HELPER(Square, float)
CLASS_TEMPLATE_SPECIFICATION_HELPER(Triangle, float)
CLASS_TEMPLATE_SPECIFICATION_HELPER(Point, float)
template std::ostream &operator <<(std::ostream &, Point<float> const &);
template RelativePos relate(Point<float> const &, Triangle<float> const &, float const &);
template RelativePos relate(Point<float> const &, Square<float> const &, float const &);
