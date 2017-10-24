#include "object/geometry/normal_triangle.hpp"

using namespace px;

BaseNormalTriangle::BaseNormalTriangle(const BaseMaterial *const &material,
                                       const Transformation *const &trans)
        : BaseGeometry(material, trans, 3)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BaseNormalTriangle::hitCheck(Ray const &ray,
                                  double const &t_start,
                                  double const &t_end,
                                  double &hit_at) const
{
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseNormalTriangle::normalVec(double const &x, double const &y, double const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<double> BaseNormalTriangle::getTextureCoord(double const &x, double const &y,
                                       double const &z) const
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
        : BaseNormalTriangle(material.get(), trans.get()),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

NormalTriangle::~NormalTriangle()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *NormalTriangle::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseNormalTriangle)));

        _material = _material_ptr == nullptr ? nullptr : _material_ptr->up2Gpu();
        _transformation = _transformation_ptr == nullptr ? nullptr : _transformation_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseNormalTriangle*>(this),
                                 sizeof(BaseNormalTriangle),
                                 cudaMemcpyHostToDevice));

        _material = _material_ptr.get();
        _transformation = _transformation_ptr.get();

        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void NormalTriangle::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_transformation_ptr.use_count() == 1)
        _transformation_ptr->clearGpuData();
    if (_material_ptr.use_count() == 1)
        _material_ptr->clearGpuData();

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}
