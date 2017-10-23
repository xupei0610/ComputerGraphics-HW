#include "object/material/checkerboard_material.hpp"

using namespace px;

BaseCheckerboardMaterial::BaseCheckerboardMaterial(Light const &ambient,
                                           Light const &diffuse,
                                           Light const &specular,
                                           int const &specular_exponent,
                                           Light const &transmissive,
                                           double const &refractive_index,
                                           double const &dim_scale,
                                           double const &color_scale,
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
Light BaseCheckerboardMaterial::getAmbient(double const &u, double const &v, double const &w) const
{
    if (((u > 0 ? std::fmod(u * _dim_scale, 1.0) > 0.5 :
          std::fmod(-u * _dim_scale, 1.0) <= 0.5) ^
         (v > 0 ? std::fmod(v * _dim_scale, 1.0) > 0.5 :
          std::fmod(-v * _dim_scale, 1.0) <= 0.5)) == 1)
        return _ambient;
    return _ambient*_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getDiffuse(double const &u, double const &v, double const &w) const
{
    if (((u > 0 ? std::fmod(u * _dim_scale, 1.0) > 0.5 :
          std::fmod(-u * _dim_scale, 1.0) <= 0.5) ^
         (v > 0 ? std::fmod(v * _dim_scale, 1.0) > 0.5 :
          std::fmod(-v * _dim_scale, 1.0) <= 0.5)) == 1)
        return _diffuse;
    return _diffuse*_color_scale;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getSpecular(double const &, double const &, double const &) const
{
    return _specular;
}

PX_CUDA_CALLABLE
int BaseCheckerboardMaterial::specularExp(double const &, double const &, double const &) const
{
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseCheckerboardMaterial::getTransmissive(double const &x, double const &y, double const &z) const
{
    return _transmissive;
}

PX_CUDA_CALLABLE
double BaseCheckerboardMaterial::refractiveIndex(double const &, double const &, double const &) const
{
    return _refractive_index;
}

std::shared_ptr<BaseMaterial> CheckerboardMaterial::create(Light const &ambient,
                                                           Light const &diffuse,
                                                           Light const &specular,
                                                           int const &specular_exponent,
                                                           Light const &transmissive,
                                                           double const &refractive_index,
                                                           double const &dim_scale,
                                                           double const &color_scale,
                                                           std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<BaseMaterial>(new CheckerboardMaterial(ambient,
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
                                           double const &refractive_index,
                                           double const &dim_scale,
                                           double const &color_scale,
                                           std::shared_ptr<BumpMapping> const &bump_mapping)
        : BaseCheckerboardMaterial(ambient, diffuse,
                                   specular, specular_exponent,
                                   transmissive, refractive_index,
                                   dim_scale, color_scale,
                                   bump_mapping.get()),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

BaseMaterial* CheckerboardMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseCheckerboardMaterial)));

        _bump_mapping = _bump_mapping_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseCheckerboardMaterial*>(this),
                                 sizeof(BaseCheckerboardMaterial),
                                 cudaMemcpyHostToDevice));

        _bump_mapping = _bump_mapping_ptr.get();
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void CheckerboardMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr))
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void CheckerboardMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setRefractiveIndex(double const &ior)
{
    _refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _bump_mapping = bm.get();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setDimScale(double const &dim_scale)
{
    _dim_scale = dim_scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void CheckerboardMaterial::setColorScale(double const &color_scale)
{
    _color_scale = color_scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
