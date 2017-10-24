#include "object/structure/bvh.hpp"

#include <cfloat>

using namespace px;

Direction const BaseBVH::SNORM_VEC[7] = {
        { 1,  0, 0},
        { 0,  1, 0},
        { 0,  0, 1},
        { 1,  1, 1},
        {-1,  1, 1},
        {-1, -1, 1},
        { 1, -1, 1}
};

BaseBVH::BaseExtent::BaseExtent(BaseGeometry *const &obj)
    : obj(obj)
{}

bool BaseBVH::BaseExtent::hitCheck(Ray const &ray,
                                   double const &range_start,
                                   double const &range_end,
                                   const double * const &num,
                                   const double * const &den) const
{
    auto tn = -FLT_MAX;
    auto tf = FLT_MAX;
    double tmp_tn;
    double tmp_tf;

#define CHECK_HIT(idx)                                          \
    tmp_tn = (lower_bound_ ## idx - num[idx]) / den[idx];       \
    tmp_tf = (upper_bound_ ## idx - num[idx]) / den[idx];       \
    if (den[idx] < 0)                                           \
    {                                                           \
        auto tmp = tmp_tn;                                      \
        tmp_tn = tmp_tf;                                        \
        tmp_tf = tmp;                                           \
    }                                                           \
    if (tmp_tn > tn) tn = tmp_tn;                               \
    if (tmp_tf < tf) tf = tmp_tf;                               \
    if (tn > tf) return false;

    CHECK_HIT(0)
    CHECK_HIT(1)
    CHECK_HIT(2)
    CHECK_HIT(3)
    CHECK_HIT(4)
    CHECK_HIT(5)
    CHECK_HIT(6)

    return true;

#undef CHECK_HIT
}

BVH::Extent::Extent(std::shared_ptr<BaseGeometry> const &obj)
        : BaseExtent(obj.get()), obj_ptr(obj)
{
    auto n_vert = 0;
    auto vert = obj->rawVertices(n_vert);
#define SET_BOUND(idx)                                          \
    for (auto j = 0; j < n_vert; ++j)                           \
    {                                                           \
        auto b = SNORM_VEC[idx].dot(vert[j]);                   \
        if (b < lower_bound_ ## idx) lower_bound_ ## idx = b;   \
        if (b > upper_bound_ ## idx) upper_bound_ ## idx = b;   \
    }
    SET_BOUND(0)
    SET_BOUND(1)
    SET_BOUND(2)
    SET_BOUND(3)
    SET_BOUND(4)
    SET_BOUND(5)
    SET_BOUND(6)

#undef SET_BOUND
}

BVH::Extent::~Extent()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseBVH::BaseExtent* BVH::Extent::up2Gpu()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseBVH::BaseExtent)));

    obj = obj_ptr->up2Gpu();

    PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                             dynamic_cast<BaseBVH::BaseExtent *>(this),
                             sizeof(BaseBVH::BaseExtent),
                             cudaMemcpyHostToDevice));
    obj = obj_ptr.get();
    return _dev_ptr;
#else
    return this;
#endif
}

void BVH::Extent::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;
    if (obj_ptr.use_count() == 1)
        obj_ptr->clearGpuData();
    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
#endif
}


BaseBVH::BaseBVH()
        : BaseGeometry(nullptr, nullptr, 0),
          NORM_VEC_0(1, 0, 0),
          NORM_VEC_1(0, 1, 0),
          NORM_VEC_2(0, 0, 1),
          NORM_VEC_3(1, 1, 1),
          NORM_VEC_4(-1, 1, 1),
          NORM_VEC_5(-1, -1, 1),
          NORM_VEC_6(1, -1, 1)

{}

PX_CUDA_CALLABLE
const BaseGeometry *BaseBVH::hitCheck(Ray const &ray,
                                double const &t_start,
                                double const &t_end,
                                double &hit_at) const
{
    double num[7];
    double den[7];
#define COMPUTE_NUM_DEN(idx)                          \
    num[idx] = ray.original.dot(NORM_VEC_ ##idx);     \
    den[idx] = ray.direction.dot(NORM_VEC_ ##idx);

    COMPUTE_NUM_DEN(0)
    COMPUTE_NUM_DEN(1)
    COMPUTE_NUM_DEN(2)
    COMPUTE_NUM_DEN(3)
    COMPUTE_NUM_DEN(4)
    COMPUTE_NUM_DEN(5)
    COMPUTE_NUM_DEN(6)

#undef COMPUTE_NUM_DEN

    const BaseGeometry *obj = nullptr;
    double t;
    double end_range = t_end;
    for (auto i = 0; i < _n_exts; ++i)
    {
        if (_exts[i]->hitCheck(ray, t_start, end_range, num, den))
        {
            auto tmp = _exts[i]->obj->hit(ray, t_start, end_range, t);
            if (tmp != nullptr)
            {
                end_range = t;
                obj = tmp;
            }
        }
    }

    return obj == nullptr ? nullptr : (hit_at = end_range,  obj);
}

PX_CUDA_CALLABLE
Direction BaseBVH::normalVec(double const &x, double const &y, double const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<double> BaseBVH::getTextureCoord(double const &x, double const &y,
                                      double const &z) const
{
    return {};
}

BVH::BVH()
        : BaseBVH()
{}

BVH::~BVH()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *BVH::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseBVH)));

        auto i = 0;
        BaseExtent *gpu_exts[_n_exts];
        auto node = _extents.start;
        while (node != nullptr)
        {
            gpu_exts[i++] = node->data->up2Gpu();
            node = node->next;
        }


        PX_CUDA_CHECK(cudaMalloc(&_exts, sizeof(BaseBVH::BaseExtent*)*_n_exts));
        PX_CUDA_CHECK(cudaMemcpy(_exts, gpu_exts,
                                 sizeof(BaseBVH::BaseExtent*)*_n_exts,
                                 cudaMemcpyHostToDevice));

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseBVH*>(this),
                                 sizeof(BaseBVH),
                                 cudaMemcpyHostToDevice));

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void BVH::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    auto node = _extents.start;
    while (node != nullptr)
    {
        node->data->clearGpuData();
        node = node->next;
    }

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _exts = nullptr;
    _need_upload = true;
#endif
}

void BVH::addObj(std::shared_ptr<BaseGeometry> const &obj)
{
    _extents.add(new Extent(obj));
    _n_exts++;

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
const BaseGeometry *BVH::hitCheck(Ray const &ray,
                            double const &t_start,
                            double const &t_end,
                            double &hit_at) const
{
    double num[7];
    double den[7];

#define GET_NUM_DEN(idx)                            \
    num[idx] = ray.original.dot(NORM_VEC_ ## idx);  \
    den[idx] = ray.direction.dot(NORM_VEC_ ## idx);

    GET_NUM_DEN(0)
    GET_NUM_DEN(1)
    GET_NUM_DEN(2)
    GET_NUM_DEN(3)
    GET_NUM_DEN(4)
    GET_NUM_DEN(5)
    GET_NUM_DEN(6)

#undef GET_NUM_DEN

    const BaseGeometry *obj = nullptr;
    double t;
    double end_range = t_end;

    auto node = _extents.start;
    while (node != nullptr)
    {
        if (node->data->hitCheck(ray, t_start, end_range, num, den))
        {
            auto tmp = node->data->obj->hit(ray, t_start, end_range, t);
            if (tmp != nullptr)
            {
                end_range = t;
                obj = tmp;
            }
        }
        node = node->next;
    }

    return obj == nullptr ? nullptr : (hit_at = end_range,  obj);
}
