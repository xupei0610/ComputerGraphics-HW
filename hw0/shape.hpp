#ifndef SHAPE_HPP
#define SHAPE_HPP

#include "global.hpp"

#include <vector>
#include <cmath>
#include <iostream>

template<typename T>
class BasicShape;
template<typename T>
class Square;
template<typename T>
class Point;

#define SHAPE_PROPERTY int
#define SQUARE_SHAPE   1
#define TRIANGLE_SHAPE 2

template<typename T>
class BasicShape
{
public:
    constexpr static T PI = std::acos(-1);

    std::vector<T> const &vertices;

    T x; // center point
    T y; // center point
    T size; // distance from center to the closest border/vertex
    T angle; // rotation angle w.r.t the center point

    std::size_t num_vertices;

    virtual void rotate(Point<T> const & pos, Point<T> const &ref_pos);
    virtual void rotate(T const &ang);
    virtual void move(T const &offset_x, T const &offset_y);
    virtual void scaleTo(T const &new_size);
    virtual void scale(T const &scale);
    virtual constexpr SHAPE_PROPERTY getProperty() const { return -1; }
    virtual void updateVertices(T const &aspect_ratio) = 0;
    virtual void reset();

protected:
    BasicShape(std::size_t const &num_vertices,
               T const &x,
               T const &y,
               T const &size,
               T const &angle);
    std::vector<T> vert_data;

    T org_x;
    T org_y;
    T org_size;
    T org_angle;

public:
    virtual ~BasicShape() = default;
    DISABLE_DEFAULT_CONSTRUCTOR(BasicShape)
};

template<typename T>
class Square : public BasicShape<T>
{
public:
    constexpr static std::size_t NUM_VERTICES = 4;

    constexpr SHAPE_PROPERTY getProperty() const override
    {
        return SQUARE_SHAPE;
    }
    void updateVertices(T const &aspect_ratio) override;

    static BasicShape<T> *create(T const &x,
                                 T const &y,
                                 T const &size,
                                 T const &angle);

    Square(T const &x, T const &y, T const &size, T const &angle);
    ~Square() = default;

public:
    ENABLE_DEFAULT_CONSTRUCTOR(Square)
};

template<typename T>
class Triangle : public BasicShape<T>
{
public:
    constexpr static std::size_t NUM_VERTICES = 3;
    constexpr static T SQRT_3 = std::sqrt(3);

    constexpr SHAPE_PROPERTY getProperty() const override
    {
        return TRIANGLE_SHAPE;
    }
    void updateVertices(T const &aspect_ratio) override;

    static BasicShape<T> *create(T const &x,
                                 T const &y,
                                 T const &size,
                                 T const &angle);

    Triangle(T const &x, T const &y, T const &size, T const &angle);
    ~Triangle() = default;

public:
    ENABLE_DEFAULT_CONSTRUCTOR(Triangle)
};

enum class RelativePos
{
    Inner,
    Outer,
    Corner,
    Border
};

template<typename T>
class Point
{
public:
    T x;
    T y;

    constexpr static T CLOSE_THRESHOLD = 0.015;

    virtual RelativePos relativeTo(const BasicShape<T> * const &shape) const;

    Point();
    Point(T const &x, T const &y);
    ~Point() = default;
    ENABLE_DEFAULT_CONSTRUCTOR(Point)
};

template<typename T>
RelativePos relate(Point<T> const &point, Square<T> const &square);
template<typename T>
RelativePos relate(Point<T> const &point, Triangle<T> const &triangle);

template<typename T>
std::ostream &operator <<(std::ostream &os, Point<T> const &p);

#endif // SHAPE_HPP
