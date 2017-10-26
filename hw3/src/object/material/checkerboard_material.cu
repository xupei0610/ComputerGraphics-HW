#include "object/material/checkerboard_material.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

BaseCheckerboardMaterial::BaseCheckerboardMaterial(Light const &ambient,
                                           Light const &diffuse,
                                           Light const &specular,
                                           int const &specular_exponent,
                                           Light const &transmissive,
                                           PREC const &refractive_index,
                                           PREC const &dim_scale,
                                           PREC const &color_scale,
                                           const BumpMapping * const &bump_mapping)
        : BaseMaterial(bump_mapping),
          _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index),
          _dim_scale(dim_scale),
          _color_scale(color_scale)
{}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    if (((u > 0 ? std::fmod(u * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-u * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) <= 0.5) ^
         (v > 0 ? std::fmod(v * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-v * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) <= 0.5)) == 1)
        return _ambient;
    return _ambient*_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    if (((u > 0 ? std::fmod(u * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-u * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) <= 0.5) ^
         (v > 0 ? std::fmod(v * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) > 0.5 :
          std::fmod(-v * _dim_scale, static_cast<decltype(_dim_scale)>(1.0)) <= 0.5)) == 1)
        return _diffuse;
    return _diffuse*_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getSpecular(PREC const &, PREC const &, PREC const &) const
{
    return _specular;
}

PX_CUDA_CALLABLE
int BaseCheckerboardMaterial::specularExp(PREC const &, PREC const &, PREC const &) const
{
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getTransmissive(PREC const &x, PREC const &y, PREC const &z) const
{
    return _transmissive;
}

PX_CUDA_CALLABLE
PREC BaseCheckerboardMaterial::refractiveIndex(PREC const &, PREC const &, PREC const &) const
{
    return _refractive_index;
}

std::shared_ptr<Material> CheckerboardMaterial::create(Light const &ambient,
                                                           Light const &diffuse,
                                                           Light const &specular,
                                                           int const &specular_exponent,
                                                           Light const &transmissive,
                                                           PREC const &refractive_index,
                                                           PREC const &dim_scale,
                                                           PREC const &color_scale,
                                                           std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<Material>(new CheckerboardMaterial(ambient,
                                                                  diffuse,
                                                                  specular,
                                                                  specular_exponent,
                                                                  transmissive,
                                                                  refractive_index,
                                                                  dim_scale,
                                                                  color_scale,
                                                                  bump_mapping));
}

CheckerboardMaterial::CheckerboardMaterial(Light const &ambient,
                                           Light const &diffuse,
                                           Light const &specular,
                                           int const &specular_exponent,
                                           Light const &transmissive,
                                           PREC const &refractive_index,
                                           PREC const &dim_scale,
                                           PREC const &color_scale,
                                           std::shared_ptr<BumpMapping> const &bump_mapping)
        : _obj(new BaseCheckerboardMaterial(ambient, diffuse,
                                   specular, specular_exponent,
                                   transmissive, refractive_index,
                                   dim_scale, color_scale,
                                   bump_mapping.get())),
          _base_obj(_obj),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

CheckerboardMaterial::~CheckerboardMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseMaterial *const &CheckerboardMaterial::obj() const noexcept
{
    return _base_obj;
}

BaseMaterial** CheckerboardMaterial::devPtr()
{
    return _dev_ptr;
}

void CheckerboardMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseMaterial**)));

        if (_bump_mapping_ptr != nullptr)
            _bump_mapping_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::CheckerboardMaterial(_dev_ptr,
                                         _obj->_ambient, _obj->_diffuse,
                                         _obj->_specular, _obj->_specular_exponent,
                                         _obj->_transmissive, _obj->_refractive_index,
                                         _obj->_dim_scale, _obj->_color_scale,
                                         _bump_mapping_ptr == nullptr ? nullptr :_bump_mapping_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void CheckerboardMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_bump_mapping_ptr.use_count() == 1)
        _bump_mapping_ptr->clearGpuData();

    GpuCreator::destroy(_dev_ptr);

    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void CheckerboardMaterial::setAmbient(Light const &ambient)
{
    _obj->_ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDiffuse(Light const &diffuse)
{
    _obj->_diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setSpecular(Light const &specular)
{
    _obj->_specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->_specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setTransmissive(Light const &transmissive)
{
    _obj->_transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->_refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _obj->setBumpMapping(bm.get());
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDimScale(PREC const &dim_scale)
{
    _obj->_dim_scale = dim_scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setColorScale(PREC const &color_scale)
{
    _obj->_color_scale = color_scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
