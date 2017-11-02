#include "object/material/brick_material.hpp"

using namespace px;

BaseBrickMaterial::BaseBrickMaterial(Light const &ambient,
                                     Light const &diffuse,
                                     Light const &specular,
                                     int const &specular_exponent,
                                     Light const &transmissive,
                                     PREC const &refractive_index,
                                     Light const &ambient_edge,
                                     Light const &diffuse_edge,
                                     Light const &specular_edge,
                                     int const &specular_exponent_edge,
                                     Light const &transmissive_edge,
                                     PREC const &refractive_index_edge,
                                     PREC const &scale,
                                     PREC const &edge_width,
                                     PREC const &edge_height)
        : _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index),
          _ambient_edge(ambient_edge),
          _diffuse_edge(diffuse_edge),
          _specular_edge(specular_edge),
          _specular_exponent_edge(specular_exponent_edge),
          _transmissive_edge(transmissive_edge),
          _refractive_index_edge(refractive_index_edge),
          _scale(scale),
          _edge_width(edge_width),
          _edge_height(edge_height)
{}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getAmbient(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_ambient_edge;
    return o->_ambient;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getDiffuse(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_diffuse_edge;
    return o->_diffuse;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getSpecular(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_specular_edge;
    return o->_specular;
}

PX_CUDA_CALLABLE
int BaseBrickMaterial::getSpecularExp(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_specular_exponent_edge;
    return o->_specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getTransmissive(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_transmissive_edge;
    return o->_transmissive;
}

PX_CUDA_CALLABLE
PREC BaseBrickMaterial::getRefractiveIndex(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseBrickMaterial*>(obj);
    if (o->onEdge(u, v, w))
        return o->_refractive_index_edge;
    return o->_refractive_index;
}

PX_CUDA_CALLABLE
bool BaseBrickMaterial::onEdge(PREC const &u,
                               PREC const &v,
                               PREC const &w) const noexcept
{
    auto tx = static_cast<int>(std::floor(_scale*u));
    auto ty = static_cast<int>(std::floor(_scale*v));
    return ((std::abs(_scale*u - tx) < _edge_width) && ((tx & 0x0001) == (ty & 0x0001))) || (std::abs(_scale*v - ty) < _edge_height);
}


void BaseBrickMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
}
void BaseBrickMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
}
void BaseBrickMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
}
void BaseBrickMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
}
void BaseBrickMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
}
void BaseBrickMaterial::setRefractiveIndex(PREC const &ior)
{
    _refractive_index = ior;
}
void BaseBrickMaterial::setAmbientEdge(Light const &ambient)
{
    _ambient_edge = ambient;
}
void BaseBrickMaterial::setDiffuseEdge(Light const &diffuse)
{
    _diffuse_edge = diffuse;
}
void BaseBrickMaterial::setSpecularEdge(Light const &specular)
{
    _specular_edge = specular;
}
void BaseBrickMaterial::setSpecularExpEdge(int const &specular_exp)
{
    _specular_exponent_edge = specular_exp;
}
void BaseBrickMaterial::setTransmissiveEdge(Light const &transmissive)
{
    _transmissive_edge = transmissive;
}
void BaseBrickMaterial::setRefractiveIndexEdge(PREC const &ior)
{
    _refractive_index_edge = ior;
}
void BaseBrickMaterial::setScale(PREC const &scale)
{
    _scale = scale;
}
void BaseBrickMaterial::setEdgeWidth(PREC const &edge_width)
{
    _edge_width = edge_width;
}
void BaseBrickMaterial::setEdgeHeight(PREC const &edge_height)
{
    _edge_height = edge_height;
}

std::shared_ptr<BaseMaterial> BrickMaterial::create(Light const &ambient,
                                                    Light const &diffuse,
                                                    Light const &specular,
                                                    int const &specular_exponent,
                                                    Light const &transmissive,
                                                    PREC const &refractive_index,
                                                    Light const &ambient_edge,
                                                    Light const &diffuse_edge,
                                                    Light const &specular_edge,
                                                    int const &specular_exponent_edge,
                                                    Light const &transmissive_edge,
                                                    PREC const &refractive_index_edge,
                                                    PREC const &scale,
                                                    PREC const &edge_width,
                                                    PREC const &edge_height)
{
    return std::shared_ptr<BaseMaterial>(new BrickMaterial(ambient,
                                                           diffuse,
                                                           specular,
                                                           specular_exponent,
                                                           transmissive,
                                                           refractive_index,
                                                           ambient_edge,
                                                           diffuse_edge,
                                                           specular_edge,
                                                           specular_exponent_edge,
                                                           transmissive_edge,
                                                           refractive_index_edge,
                                                           scale,
                                                           edge_width,
                                                           edge_height));
}

BrickMaterial::BrickMaterial(Light const &ambient,
                             Light const &diffuse,
                             Light const &specular,
                             int const &specular_exponent,
                             Light const &transmissive,
                             PREC const &refractive_index,
                             Light const &ambient_edge,
                             Light const &diffuse_edge,
                             Light const &specular_edge,
                             int const &specular_exponent_edge,
                             Light const &transmissive_edge,
                             PREC const &refractive_index_edge,
                             PREC const &scale,
                             PREC const &edge_width,
                             PREC const &edge_height)
        : BaseMaterial(),
          _obj(new BaseBrickMaterial(ambient, diffuse,
                            specular, specular_exponent,
                            transmissive, refractive_index,
                            ambient_edge, diffuse_edge,
                            specular_edge, specular_exponent_edge,
                            transmissive_edge, refractive_index_edge,
                            scale, edge_width, edge_height)),
          _gpu_obj(nullptr),
          _need_upload(true)
{}

BrickMaterial::~BrickMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnAmbient_t __fn_ambient_brick_material = BaseBrickMaterial::getAmbient;
__device__ fnDiffuse_t __fn_diffuse_brick_material = BaseBrickMaterial::getDiffuse;
__device__ fnSpecular_t __fn_specular_brick_material = BaseBrickMaterial::getSpecular;
__device__ fnSpecularExp_t __fn_specular_exp_brick_material = BaseBrickMaterial::getSpecularExp;
__device__ fnTransmissive_t __fn_transmissive_brick_material = BaseBrickMaterial::getTransmissive;
__device__ fnRefractiveIndex_t __fn_refractive_index_brick_material = BaseBrickMaterial::getRefractiveIndex;
#endif

void BrickMaterial::up2Gpu()
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
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseBrickMaterial)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(MaterialObj)));
        }
        if (fn_ambient_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_ambient_h, __fn_ambient_brick_material, sizeof(fnAmbient_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_diffuse_h, __fn_diffuse_brick_material, sizeof(fnDiffuse_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_h, __fn_specular_brick_material, sizeof(fnSpecular_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_exp_h, __fn_specular_exp_brick_material, sizeof(fnSpecularExp_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_transmissive_h, __fn_transmissive_brick_material, sizeof(fnTransmissive_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_refractive_index_h, __fn_refractive_index_brick_material, sizeof(fnRefractiveIndex_t)));
        }
        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseBrickMaterial),
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

void BrickMaterial::clearGpuData()
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

int BrickMaterial::specularExp(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getSpecularExp(_obj, u, v, w);
}
PREC BrickMaterial::refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getRefractiveIndex(_obj, u, v, w);
}
Light BrickMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getAmbient(_obj, u, v, w);
}
Light BrickMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getDiffuse(_obj, u, v, w);
}
Light BrickMaterial::getSpecular(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getSpecular(_obj, u, v, w);
}
Light BrickMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseBrickMaterial::getTransmissive(_obj, u, v, w);
}

void BrickMaterial::setAmbient(Light const &ambient)
{
    _obj->setAmbient(ambient);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuse(Light const &diffuse)
{
    _obj->setDiffuse(diffuse);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecular(Light const &specular)
{
    _obj->setSpecular(specular);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->setSpecularExp(specular_exp);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissive(Light const &transmissive)
{
    _obj->setTransmissive(transmissive);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->setRefractiveIndex(ior);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setAmbientEdge(Light const &ambient)
{
    _obj->setAmbientEdge(ambient);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuseEdge(Light const &diffuse)
{
    _obj->setDiffuseEdge(diffuse);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularEdge(Light const &specular)
{
    _obj->setSpecularEdge(specular);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExpEdge(int const &specular_exp)
{
    _obj->setSpecularExpEdge(specular_exp);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissiveEdge(Light const &transmissive)
{
    _obj->setTransmissiveEdge(transmissive);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndexEdge(PREC const &ior)
{
    _obj->setRefractiveIndexEdge(ior);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setScale(PREC const &scale)
{
    _obj->setScale(scale);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeWidth(PREC const &edge_width)
{
    _obj->setEdgeWidth(edge_width);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeHeight(PREC const &edge_height)
{
    _obj->setEdgeHeight(edge_height);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
