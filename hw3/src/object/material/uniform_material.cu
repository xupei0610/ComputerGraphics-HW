#include "object/material/uniform_material.hpp"

using namespace px;

BaseUniformMaterial::BaseUniformMaterial(Light const &ambient,
                                         Light const &diffuse,
                                         Light const &specular,
                                         int const &specular_exponent,
                                         Light const &transmissive,
                                         PREC const &refractive_index)
        : _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index)
{}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getAmbient(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_ambient;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getDiffuse(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_diffuse;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getSpecular(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_specular;
}

PX_CUDA_CALLABLE
int BaseUniformMaterial::getSpecularExp(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseUniformMaterial::getTransmissive(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_transmissive;
}

PX_CUDA_CALLABLE
PREC BaseUniformMaterial::getRefractiveIndex(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseUniformMaterial*>(obj)->_refractive_index;
}

void BaseUniformMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
}
void BaseUniformMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
}
void BaseUniformMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
}
void BaseUniformMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
}
void BaseUniformMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
}
void BaseUniformMaterial::setRefractiveIndex(PREC const &ior)
{
    _refractive_index = ior;
}

std::shared_ptr<BaseMaterial> UniformMaterial::create(Light const &ambient,
                                                Light const &diffuse,
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive,
                                                PREC const &refractive_index)
{
    return std::shared_ptr<BaseMaterial>(new UniformMaterial(ambient,
                                                       diffuse,
                                                       specular,
                                                       specular_exponent,
                                                       transmissive,
                                                       refractive_index));
}

UniformMaterial::UniformMaterial(Light const &ambient,
                             Light const &diffuse,
                             Light const &specular,
                             int const &specular_exponent,
                             Light const &transmissive,
                             PREC const &refractive_index)
        : BaseMaterial(),
          _obj(new BaseUniformMaterial(ambient, diffuse,
                                       specular, specular_exponent,
                                       transmissive, refractive_index)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

UniformMaterial::~UniformMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnAmbient_t __fn_ambient_uniform_material = BaseUniformMaterial::getAmbient;
__device__ fnDiffuse_t __fn_diffuse_uniform_material = BaseUniformMaterial::getDiffuse;
__device__ fnSpecular_t __fn_specular_uniform_material = BaseUniformMaterial::getSpecular;
__device__ fnSpecularExp_t __fn_specular_exp_uniform_material = BaseUniformMaterial::getSpecularExp;
__device__ fnTransmissive_t __fn_transmissive_uniform_material = BaseUniformMaterial::getTransmissive;
__device__ fnRefractiveIndex_t __fn_refractive_index_uniform_material = BaseUniformMaterial::getRefractiveIndex;
#endif

void UniformMaterial::up2Gpu()
{
#ifdef USE_CUDA
    static fnAmbient_t fn_ambient_h = nullptr;
    static fnDiffuse_t fn_diffuse_h;
    static fnSpecular_t fn_specular_h;
    static fnSpecularExp_t fn_specular_exp_h;
    static fnTransmissive_t fn_transmissive_h;
    static fnRefractiveIndex_t fn_refractive_index_h;
    
    if (_need_upload)
    {
        if (dev_ptr == nullptr)
        {
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseUniformMaterial)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(MaterialObj)));
        }
        if (fn_ambient_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_ambient_h, __fn_ambient_uniform_material, sizeof(fnAmbient_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_diffuse_h, __fn_diffuse_uniform_material, sizeof(fnDiffuse_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_h, __fn_specular_uniform_material, sizeof(fnSpecular_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_exp_h, __fn_specular_exp_uniform_material, sizeof(fnSpecularExp_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_transmissive_h, __fn_transmissive_uniform_material, sizeof(fnTransmissive_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_refractive_index_h, __fn_refractive_index_uniform_material, sizeof(fnRefractiveIndex_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseUniformMaterial),
                                 cudaMemcpyHostToDevice));
        MaterialObj tmp(_gpu_obj,
                        fn_ambient_h, fn_diffuse_h,
                        fn_specular_h, fn_specular_exp_h,
                        fn_transmissive_h, fn_refractive_index_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(MaterialObj),
                                 cudaMemcpyHostToDevice));
        _need_upload = false;
    }
#endif
}

void UniformMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_gpu_obj != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_obj = nullptr;
    }
    BaseMaterial::clearGpuData();
#endif
}

int UniformMaterial::specularExp(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getSpecularExp(_obj, u, v, w);
}
PREC UniformMaterial::refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getRefractiveIndex(_obj, u, v, w);
}
Light UniformMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getAmbient(_obj, u, v, w);
}
Light UniformMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getDiffuse(_obj, u, v, w);
}
Light UniformMaterial::getSpecular(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getSpecular(_obj, u, v, w);
}
Light UniformMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseUniformMaterial::getTransmissive(_obj, u, v, w);
}

void UniformMaterial::setAmbient(Light const &ambient)
{
    _obj->setAmbient(ambient);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setDiffuse(Light const &diffuse)
{
    _obj->setDiffuse(diffuse);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecular(Light const &specular)
{
    _obj->setSpecular(specular);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->setSpecularExp(specular_exp);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setTransmissive(Light const &transmissive)
{
    _obj->setTransmissive(transmissive);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void UniformMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->setRefractiveIndex(ior);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
