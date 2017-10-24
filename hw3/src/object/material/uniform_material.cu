#include "object/material/uniform_material.hpp"

using namespace px;

BaseUniformMaterial::BaseUniformMaterial(Light const &ambient,
                                         Light const &diffuse,
                                         Light const &specular,
                                         int const &specular_exponent,
                                         Light const &transmissive,
                                         double const &refractive_index,
                                         const BumpMapping * const &bump_mapping)
        : BaseMaterial(bump_mapping),
          _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index)
{}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getAmbient(double const &, double const &, double const &) const
{
    return _ambient;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getDiffuse(double const &, double const &, double const &) const
{
    return _diffuse;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getSpecular(double const &, double const &, double const &) const
{
    return _specular;
}

PX_CUDA_CALLABLE
int BaseUniformMaterial::specularExp(double const &, double const &, double const &) const
{
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getTransmissive(double const &x, double const &y, double const &z) const
{
    return _transmissive;
}

PX_CUDA_CALLABLE
double BaseUniformMaterial::refractiveIndex(double const &, double const &, double const &) const
{
    return _refractive_index;
}

std::shared_ptr<BaseMaterial> UniformMaterial::create(Light const &ambient,
                                                      Light const &diffuse,
                                                      Light const &specular,
                                                      int const &specular_exponent,
                                                      Light const &transmissive,
                                                      double const &refractive_index,
                                                      std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<BaseMaterial>(new UniformMaterial(ambient,
                                                             diffuse,
                                                             specular,
                                                             specular_exponent,
                                                             transmissive,
                                                             refractive_index,
                                                             bump_mapping));
}

UniformMaterial::UniformMaterial(Light const &ambient,
                                 Light const &diffuse,
                                 Light const &specular,
                                 int const &specular_exponent,
                                 Light const &transmissive,
                                 double const &refractive_index,
                                 std::shared_ptr<BumpMapping> const &bump_mapping)
        : BaseUniformMaterial(ambient,
                              diffuse,
                              specular, specular_exponent,
                              transmissive, refractive_index,
                              bump_mapping.get()),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

UniformMaterial::~UniformMaterial()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseMaterial* UniformMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseUniformMaterial)));

        _bump_mapping = _bump_mapping_ptr->up2Gpu();
        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseUniformMaterial*>(this),
                                 sizeof(BaseUniformMaterial),
                                 cudaMemcpyHostToDevice));

        _bump_mapping = _bump_mapping_ptr.get();
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void UniformMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_bump_mapping_ptr.use_count() == 1)
        _bump_mapping_ptr->clearGpu();

    PX_CUDA_CHECK(cudaFree(_dev_ptr));
    _dev_ptr = nullptr;
    _need_upload = true;
#endif

}

void UniformMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setRefractiveIndex(double const &ior)
{
    _refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _bump_mapping = bm.get();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
