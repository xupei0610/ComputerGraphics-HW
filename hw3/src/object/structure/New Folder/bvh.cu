#include "object/structure/bvh.hpp"

#include <cfloat>

using namespace px;

//Direction const BaseBVH::NORM_VEC_0 = { 1,  0, 0};
//Direction const BaseBVH::NORM_VEC_1 = { 0,  1, 0};
//Direction const BaseBVH::NORM_VEC_2 = { 0,  0, 1};
//Direction const BaseBVH::NORM_VEC_3 = { 1,  1, 1};
//Direction const BaseBVH::NORM_VEC_4 = {-1,  1, 1};
//Direction const BaseBVH::NORM_VEC_5 = {-1, -1, 1};
//Direction const BaseBVH::NORM_VEC_6 = { 1, -1, 1};
//Direction const BaseBVH::NORM_VEC ={
//        { 1,  0, 0},
//        { 0,  1, 0},
//        { 0,  0, 1},
//        { 1,  1, 1},
//        {-1,  1, 1},
//        {-1, -1, 1},
//        { 1, -1, 1}
//};

BaseBVH::Extent::Extent(BaseGeometry *const &obj)
    : obj(obj)
{
/**
//    auto n_vert = 0;
//    auto vert = obj->rawVertices(n_vert);
//#define SET_BOUND(idx)                                          \
//    for (auto j = 0; j < n_vert; ++j)                           \
//    {                                                           \
//        auto b = NORM_VEC_ ## idx ## .dot(vert[j]);             \
//        if (b < lower_bound_ ## idx) lower_bound_ ## idx = b;   \
//        if (b > upper_bound_ ## idx) upper_bound_ ## idx = b;   \
//    }
//    SET_BOUND(0)
//    SET_BOUND(1)
//    SET_BOUND(2)
//    SET_BOUND(3)
//    SET_BOUND(4)
//    SET_BOUND(5)
//    SET_BOUND(6)
//
//#undef SET_BOUND
**/
}

bool BaseBVH::Extent::hitCheck(Ray const &ray,
                                   PREC const &range_start,
                                   PREC const &range_end,
                                   const PREC * const &num,
                                   const PREC * const &den) const
{
    auto tn = -FLT_MAX;
    auto tf = FLT_MAX;
    PREC tmp_tn;
    PREC tmp_tf;

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

PX_CUDA_CALLABLE
BaseBVH::BaseBVH()
        : BaseGeometry(nullptr, nullptr, 0)
{}

PX_CUDA_CALLABLE
const BaseGeometry *BaseBVH::hitCheck(Ray const &ray,
                                PREC const &t_start,
                                PREC const &t_end,
                                PREC &hit_at) const
{
/**
//    PREC num[7];
//    PREC den[7];
//
//#define GET_NUM_DEN(idx)                            \
//    num[idx] = original.dot(NORM_VEC_ ##idx);  \
//    den[idx] = direction.dot(NORM_VEC_ ##idx);
//
//    GET_NUM_DEN(0)
//    GET_NUM_DEN(1)
//    GET_NUM_DEN(2)
//    GET_NUM_DEN(3)
//    GET_NUM_DEN(4)
//    GET_NUM_DEN(5)
//    GET_NUM_DEN(6)
//
//#undef GET_NUM_DEN
 **/

    const BaseGeometry *obj = nullptr;
    PREC t;
    PREC end_range = t_end;

    auto node = _extents.start;
    while (node != nullptr)
    {
//        if (node->data->hitCheck(ray, t_start, end_range, num, den))
//        {
            auto tmp = node->data->obj->hit(ray, t_start, end_range, t);
            if (tmp != nullptr)
            {
                end_range = t;
                obj = tmp;
            }
//        }
        node = node->next;
    }

    return obj == nullptr ? nullptr : (hit_at = end_range,  obj);
}

PX_CUDA_CALLABLE
Direction BaseBVH::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseBVH::getTextureCoord(PREC const &x, PREC const &y,
                                      PREC const &z) const
{
    return {};
}

BVH::BVH()
        : _obj(new BaseBVH()), _base_obj(_obj),
          _dev_ptr(nullptr), _need_upload(true)
{}

BVH::~BVH()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &BVH::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **BVH::devPtr()
{
    return _dev_ptr;
}

void BVH::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseGeometry**)));


        for (auto &o : _objects_ptr)
            o->up2Gpu();

        auto i = 0;
        BaseGeometry **gpu_objs[_obj->_extents.n];
        for (auto &o : _objects_ptr)
            gpu_objs[i++] = o->devPtr();

        BaseGeometry ***tmp;

        PX_CUDA_CHECK(cudaMalloc(&tmp, sizeof(BaseGeometry **) * _obj->_extents.n));
        PX_CUDA_CHECK(cudaMemcpy(tmp, gpu_objs, sizeof(BaseGeometry **) * _obj->_extents.n,
                                 cudaMemcpyHostToDevice));

        GpuCreator::BVH(_dev_ptr, tmp, _obj->_extents.n);

        _need_upload = false;
    }
#endif
}

void BVH::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void BVH::addObj(std::shared_ptr<Geometry> const &obj)
{
    if (obj == nullptr)
        return;

    _objects_ptr.insert(obj);
    _obj->addObj(obj->obj());

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
void BaseBVH::addObj(BaseGeometry *const &obj)
{
    _extents.add(new Extent(obj));
}

