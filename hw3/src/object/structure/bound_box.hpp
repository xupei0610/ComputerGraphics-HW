#ifndef PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
#define PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP

#include "object/structure/base_structure.hpp"

namespace px
{
class BoundBox;
class BaseBoundBox;
}

class px::BaseBoundBox : public Structure
{
protected:
    PX_CUDA_CALLABLE
    bool hitBox(Ray const &ray,
                double const &t_start,
                double const &t_end) const;

    PX_CUDA_CALLABLE
    const BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) const override;
    PX_CUDA_CALLABLE
    Vec3<double> getTextureCoord(double const &x,
                                 double const &y,
                                 double const &z) const override;
    PX_CUDA_CALLABLE
    Direction normalVec(double const &x, double const &y, double const &z) const override;

    ~BaseBoundBox() = default;

protected:
    Point _vertex_min;
    Point _vertex_max;

    Point _center;
    Vec3<double> _side;

    int _n_objs;
    BaseGeometry **_objs;

    BaseBoundBox(const Transformation * const &trans);

    BaseBoundBox &operator=(BaseBoundBox const &) = delete;
    BaseBoundBox &operator=(BaseBoundBox &&) = delete;
};

class px::BoundBox : public BaseBoundBox
{
public:
    BoundBox(std::shared_ptr<Transformation> const &trans);

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    ~BoundBox();
protected:
    struct List             // a sample list structure for loop in the CPU version of hitCheck
    {                       // which have to be compatible with CUDA __device__ standard
        struct Node
        {
            BaseGeometry *data;
            Node *next;

            Node(BaseGeometry * const &obj) : data(obj), next(nullptr)
            {}
        };

        Node *start;
        Node *end;
        int n;

        List() : start(nullptr), end(nullptr), n(0)
        {}

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

        ~List() = default; // data come from shared_ptr;
    };

    List _objects;

    PX_CUDA_CALLABLE
    const BaseGeometry * hitCheck(Ray const &ray,
                            double const &range_start,
                            double const &range_end,
                            double &hit_at) const override;

    std::shared_ptr<Transformation> _transformation_ptr;
    std::unordered_set<std::shared_ptr<BaseGeometry> > _objects_ptr;

    BaseBoundBox * _dev_ptr;
    bool _need_upload;

    BoundBox &operator=(BoundBox const &) = delete;
    BoundBox &operator=(BoundBox &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BOUND_BOX_HPP
