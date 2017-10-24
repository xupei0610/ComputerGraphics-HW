#ifndef PX_CG_OBJECT_GEOMETRY_BVH_HPP
#define PX_CG_OBJECT_GEOMETRY_BVH_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
// TODO OctTree
class BVH;
class BaseBVH;
}

class px::BaseBVH : public BaseGeometry
{
protected:
    Direction const NORM_VEC_0; // not use static for gpu hitCheck
    Direction const NORM_VEC_1;
    Direction const NORM_VEC_2;
    Direction const NORM_VEC_3;
    Direction const NORM_VEC_4;
    Direction const NORM_VEC_5;
    Direction const NORM_VEC_6;

    Direction const static SNORM_VEC[7]; // for cpu object initialization

    class BaseExtent
    {
    public:
        BaseExtent(BaseGeometry * const &obj);
        PX_CUDA_CALLABLE
        bool hitCheck(Ray const &ray,
                      double const &range_start,
                      double const &range_end,
                      const double * const &num,
                      const double * const &den) const;

        double lower_bound_0;
        double lower_bound_1;
        double lower_bound_2;
        double lower_bound_3;
        double lower_bound_4;
        double lower_bound_5;
        double lower_bound_6;
        double upper_bound_0;
        double upper_bound_1;
        double upper_bound_2;
        double upper_bound_3;
        double upper_bound_4;
        double upper_bound_5;
        double upper_bound_6;

        ~BaseExtent() = default;

        BaseGeometry *obj;
    };

protected:
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

    ~BaseBVH() = default;

protected:
    int _n_exts;
    BaseExtent **_exts;

    BaseBVH();

    BaseBVH &operator=(BaseBVH const &) = delete;
    BaseBVH &operator=(BaseBVH &&) = delete;
};

class px::BVH : public BaseBVH
{
protected:
    class Extent : public BaseBVH::BaseExtent
    {
    public:
        Extent(std::shared_ptr<BaseGeometry> const &obj);

        BaseBVH::BaseExtent * up2Gpu();
        void clearGpuData();

        std::shared_ptr<BaseGeometry> obj_ptr;
        Extent * _dev_ptr;
        ~Extent();
    };

public:
    BVH();

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    // a cpu version
    PX_CUDA_CALLABLE
    const BaseGeometry *hitCheck(Ray const &ray,
                                 double const &t_start,
                                 double const &t_end,
                                 double &hit_at) const override;


    BaseGeometry *up2Gpu() override;
    void clearGpuData() override;

    ~BVH();
protected:

    struct List             // a sample list structure for loop in the CPU version of hitCheck
    {                       // which have to be compatible with CUDA __device__ standard
        struct Node
        {
            Extent *data;
            Node *next;

            Node(Extent * const &ext) : data(ext), next(nullptr)
            {}
        };

        Node *start;
        Node *end;
        int n;

        List() : start(nullptr), end(nullptr), n(0)
        {}

        void add(Extent *const &ext)
        {
            if (start == nullptr)
            {
                start = new Node(ext);
                end = start;
            }
            else
            {
                end->next = new Node(ext);
                end = end->next;
            }
            ++n;
        }

        ~List()
        {
            if (start != nullptr)
            {
                auto node = start;
                while (node != nullptr)
                {
                    delete node->data;
                    node = node->next;
                }
            }
        }
    };

    List _extents;

    BaseBVH * _dev_ptr;
    bool _need_upload;

    BVH &operator=(BVH const &) = delete;
    BVH &operator=(BVH &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BVH_HPP
