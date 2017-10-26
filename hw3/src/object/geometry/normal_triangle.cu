#include "object/geometry/normal_triangle.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

PX_CUDA_CALLABLE
BaseNormalTriangle::BaseNormalTriangle(Point const &vertex1, Direction const &normal1,
                                       Point const &vertex2, Direction const &normal2,
                                       Point const &vertex3, Direction const &normal3,
                                       const BaseMaterial *const &material,
                                       const Transformation *const &trans)
        : BaseGeometry(material, trans, 3)
{}

PX_CUDA_CALLABLE
const BaseGeometry * BaseNormalTriangle::hitCheck(Ray const &ray,
                                  PREC const &t_start,
                                  PREC const &t_end,
                                  PREC &hit_at) const
{
    return nullptr;
}

PX_CUDA_CALLABLE
Direction BaseNormalTriangle::normalVec(PREC const &x, PREC const &y, PREC const &z) const
{
    return {};
}

PX_CUDA_CALLABLE
Vec3<PREC> BaseNormalTriangle::getTextureCoord(PREC const &x, PREC const &y,
                                       PREC const &z) const
{
    return {};
}

std::shared_ptr<Geometry> NormalTriangle::create(Point const &vertex1, Direction const &normal1,
                                                     Point const &vertex2, Direction const &normal2,
                                                     Point const &vertex3, Direction const &normal3,
                                                     std::shared_ptr<Material> const &material,
                                                     std::shared_ptr<Transformation> const &trans)
{
    return std::shared_ptr<Geometry>(new NormalTriangle(vertex1, normal1,
                                                            vertex2, normal2,
                                                            vertex3, normal3,
                                                            material, trans));
}

NormalTriangle::NormalTriangle(Point const &vertex1, Direction const &normal1,
                               Point const &vertex2, Direction const &normal2,
                               Point const &vertex3, Direction const &normal3,
                               std::shared_ptr<Material> const &material,
                               std::shared_ptr<Transformation> const &trans)
        : _obj(new BaseNormalTriangle(vertex1, normal1, vertex2, normal2, vertex3, normal3,
                             material->obj(), trans.get())),
          _base_obj(_obj),
          _a(vertex1), _na(normal1), _b(vertex1), _nb(normal1), _c(vertex1), _nc(normal1),
          _material_ptr(material), _transformation_ptr(trans),
          _dev_ptr(nullptr), _need_upload(true)
{}

NormalTriangle::~NormalTriangle()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseGeometry *const &NormalTriangle::obj() const noexcept
{
    return _base_obj;
}

BaseGeometry **NormalTriangle::devPtr()
{
    return _dev_ptr;
}

void NormalTriangle::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseGeometry **)));

        if (_material_ptr != nullptr)
            _material_ptr->up2Gpu();
        if (_transformation_ptr != nullptr)
            _transformation_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::NormalTriangle(_dev_ptr, _a, _na, _b, _nb, _c, _nc,
                                   _material_ptr == nullptr ? nullptr
                                                            : _material_ptr->devPtr(),
                                   _transformation_ptr == nullptr ? nullptr
                                                                  : _transformation_ptr->devPtr());

        _need_upload = false;
    }
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

    GpuCreator::destroy(_dev_ptr);
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}
