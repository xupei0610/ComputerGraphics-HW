#include "object/material/uniform_material.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif


using namespace px;

BaseUniformMaterial::BaseUniformMaterial(Light const &ambient,
                                         Light const &diffuse,
                                         Light const &specular,
                                         int const &specular_exponent,
                                         Light const &transmissive,
                                         PREC const &refractive_index,
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
Light BaseUniformMaterial::getAmbient(PREC const &, PREC const &, PREC const &) const
{
    return _ambient;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getDiffuse(PREC const &, PREC const &, PREC const &) const
{
    return _diffuse;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getSpecular(PREC const &, PREC const &, PREC const &) const
{
    return _specular;
}

PX_CUDA_CALLABLE
int BaseUniformMaterial::specularExp(PREC const &, PREC const &, PREC const &) const
{
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getTransmissive(PREC const &x, PREC const &y, PREC const &z) const
{
    return _transmissive;
}

PX_CUDA_CALLABLE
PREC BaseUniformMaterial::refractiveIndex(PREC const &, PREC const &, PREC const &) const
{
    return _refractive_index;
}

std::shared_ptr<Material> UniformMaterial::create(Light const &ambient,
                                                      Light const &diffuse,
                                                      Light const &specular,
                                                      int const &specular_exponent,
                                                      Light const &transmissive,
                                                      PREC const &refractive_index,
                                                      std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<Material>(new UniformMaterial(ambient,
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
                                 PREC const &refractive_index,
                                 std::shared_ptr<BumpMapping> const &bump_mapping)
        : _obj(new BaseUniformMaterial(ambient,
                              diffuse,
                              specular, specular_exponent,
                              transmissive, refractive_index,
                              bump_mapping.get())),
          _base_obj(_obj),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

UniformMaterial::~UniformMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

BaseMaterial *const &UniformMaterial::obj() const noexcept
{
    return _base_obj;
}

BaseMaterial** UniformMaterial::devPtr()
{
    return _dev_ptr;
}

void UniformMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseMaterial**)));

        if (_bump_mapping_ptr != nullptr)
            _bump_mapping_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::UniformMateiral(_dev_ptr,
                                    _obj->_ambient,
                                    _obj->_diffuse,
                                    _obj->_specular, _obj->_specular_exponent,
                                    _obj->_transmissive, _obj->_refractive_index,
                                    _bump_mapping_ptr == nullptr ? nullptr : _bump_mapping_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void UniformMaterial::clearGpuData()
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

void UniformMaterial::setAmbient(Light const &ambient)
{
    _obj->_ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setDiffuse(Light const &diffuse)
{
    _obj->_diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecular(Light const &specular)
{
    _obj->_specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->_specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setTransmissive(Light const &transmissive)
{
    _obj->_transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->_refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _obj->setBumpMapping(bm.get());
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
