#ifndef PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
#define PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
class BoundBox;
class BaseBoundBox;
}

class px::BaseBoundBox : public BaseGeometry
{
protected:
    PX_CUDA_CALLABLE
    bool hitBox(Ray const &ray,
                PREC const &t_start,
                PREC const &t_end) const;

    PX_CUDA_CALLABLE
    const BaseGeometry * hitCheck(Ray const &ray,
                            PREC const &range_start,
                            PREC const &range_end,
                            PREC &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<PREC> getTextureCoord(PREC const &x,
                                 PREC const &y,
                                 PREC const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(PREC const &x, PREC const &y, PREC const &z) const override;

public:
    PX_CUDA_CALLABLE
    BaseBoundBox(const Transformation * const &trans);
    PX_CUDA_CALLABLE
    ~BaseBoundBox() = default;

    PX_CUDA_CALLABLE
    void addObj(BaseGeometry * const &obj);

protected:
    struct List
    {
        struct Node
        {
            BaseGeometry *data;
            Node *next;

            PX_CUDA_CALLABLE
            Node(BaseGeometry * const &obj) : data(obj), next(nullptr)
            {}

            PX_CUDA_CALLABLE
            ~Node() = default;
        };

        Node *start;
        Node *end;
        int n;

        PX_CUDA_CALLABLE
        List() : start(nullptr), end(nullptr), n(0)
        {}

        PX_CUDA_CALLABLE
        void add(BaseGeometry *const &obj)
        {
            if (start == nullptr)
            {
                start = new Node(obj);
                end = start;
            }
            else
            {
                end->next = new Node(obj);
                end = end->next;
            }
            ++n;
        }

        PX_CUDA_CALLABLE
        ~List() = default; // data come from shared_ptr;
    };

    List _objects;

    Point _vertex_min;
    Point _vertex_max;

    Point _center;
    Vec3<PREC> _side;

    BaseBoundBox &operator=(BaseBoundBox const &) = delete;
    BaseBoundBox &operator=(BaseBoundBox &&) = delete;

    friend class BoundBox;
};

class px::BoundBox : public Geometry
{
public:
    BoundBox(std::shared_ptr<Transformation> const &trans);

    void addObj(std::shared_ptr<Geometry> const &obj);

    BaseGeometry *const &obj() const noexcept override;
    BaseGeometry **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override;

    ~BoundBox();
protected:
    BaseBoundBox *_obj;
    BaseGeometry *_base_obj;

    std::shared_ptr<Transformation> _transformation_ptr;
    std::unordered_set<std::shared_ptr<Geometry> > _objects_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    BoundBox &operator=(BoundBox const &) = delete;
    BoundBox &operator=(BoundBox &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
