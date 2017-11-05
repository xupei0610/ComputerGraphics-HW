#ifndef PX_CG_OBJECT_GEOMETRY_BVH_HPP
#define PX_CG_OBJECT_GEOMETRY_BVH_HPP

#include "object/geometry/base_geometry.hpp"

namespace px
{
// TODO OCTree
class BVH;
class BaseBVH;
}

class px::BaseBVH : public BaseGeometry
{
protected:
    class Extent
    {
    public:
        PX_CUDA_CALLABLE
        Extent(BaseGeometry * const &obj);
        PX_CUDA_CALLABLE
        bool hitCheck(Ray const &ray,
                      PREC const &range_start,
                      PREC const &range_end,
                      const PREC * const &num,
                      const PREC * const &den) const;

        PREC lower_bound_0;
        PREC lower_bound_1;
        PREC lower_bound_2;
        PREC lower_bound_3;
        PREC lower_bound_4;
        PREC lower_bound_5;
        PREC lower_bound_6;
        PREC upper_bound_0;
        PREC upper_bound_1;
        PREC upper_bound_2;
        PREC upper_bound_3;
        PREC upper_bound_4;
        PREC upper_bound_5;
        PREC upper_bound_6;

        PX_CUDA_CALLABLE
        ~Extent() = default;

        BaseGeometry *obj; // data from shared_ptr, no need to delete
    };

    struct List             // a sample list structure for loop in the CPU version of hitCheck
    {                       // which have to be compatible with CUDA __device__ standard
        struct Node
        {
            Extent *data;
            Node *next;

            PX_CUDA_CALLABLE
            Node(Extent * const &ext) : data(ext), next(nullptr)
            {}
        };

        Node *start;
        Node *end;
        int n;

        PX_CUDA_CALLABLE
        List() : start(nullptr), end(nullptr), n(0)
        {}

        PX_CUDA_CALLABLE
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

        PX_CUDA_CALLABLE
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

public:
    PX_CUDA_CALLABLE
    void addObj(BaseGeometry *const &obj);

protected:
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
    ~BaseBVH() = default;
    PX_CUDA_CALLABLE
    BaseBVH();
protected:

    BaseBVH &operator=(BaseBVH const &) = delete;
    BaseBVH &operator=(BaseBVH &&) = delete;

    friend class BVH;
};

class px::BVH : public BaseGeometry
{
public:
    BVH();

    void addObj(std::shared_ptr<BaseGeometry> const &obj);

    void up2Gpu() override;
    void clearGpuData() override;

    ~BVH();
protected:
    BaseBVH *_obj;
    BaseGeometry *_base_obj;

    std::unordered_set<std::shared_ptr<BaseGeometry> > _objects_ptr;

    BaseGeometry **_dev_ptr;
    bool _need_upload;

    BVH &operator=(BVH const &) = delete;
    BVH &operator=(BVH &&) = delete;
};

#endif // PX_CG_OBJECT_GEOMETRY_BVH_HPP
