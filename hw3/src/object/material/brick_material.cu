#include "object/material/brick_material.hpp"

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

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
                                     PREC const &edge_height,
                                     const BumpMapping * const &bump_mapping)
        : BaseMaterial(bump_mapping),
          _ambient(ambient),
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
Light BaseBrickMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _ambient_edge;
    return _ambient;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _diffuse_edge;
    return _diffuse;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getSpecular(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _specular_edge;
    return _specular;
}

PX_CUDA_CALLABLE
int BaseBrickMaterial::specularExp(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _specular_exponent_edge;
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _transmissive_edge;
    return _transmissive;
}

PX_CUDA_CALLABLE
PREC BaseBrickMaterial::refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const
{
    if (onEdge(u, v, w))
        return _refractive_index_edge;
    return _refractive_index;
}

PX_CUDA_CALLABLE
bool BaseBrickMaterial::onEdge(PREC const &u,
                               PREC const &v,
                               PREC const &w) const noexcept
{
    auto tx = static_cast<int>(u < 0 ? std::ceil(_scale*u) : std::floor(_scale*u));
    auto ty = static_cast<int>(v < 0 ? std::ceil(_scale*v) : std::floor(_scale*v));
    return ((std::abs(_scale*u - tx) < _edge_width) && ((tx & 0x0001) == (ty & 0x0001))) || (std::abs(_scale*v - ty) < _edge_height);
}

std::shared_ptr<Material> BrickMaterial::create(Light const &ambient,
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
                                                    PREC const &edge_height,
                                                    std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<Material>(new BrickMaterial(ambient,
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
                                                           edge_height,
                                                           bump_mapping));
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
                             PREC const &edge_height,
                             std::shared_ptr<BumpMapping> const &bump_mapping)
        : _obj(new BaseBrickMaterial(ambient, diffuse,
                            specular, specular_exponent,
                            transmissive, refractive_index,
                            ambient_edge, diffuse_edge,
                            specular_edge, specular_exponent_edge,
                            transmissive_edge, refractive_index_edge,
                            scale, edge_width, edge_height,
                            bump_mapping.get())),
          _base_obj(_obj),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

BrickMaterial::~BrickMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}
BaseMaterial *const &BrickMaterial::obj() const noexcept
{
    return _base_obj;
}

BaseMaterial **BrickMaterial::devPtr()
{
    return _dev_ptr;
}

void BrickMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseMaterial**)));

        if (_bump_mapping_ptr != nullptr)
            _bump_mapping_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::BrickMaterial(_dev_ptr,
                                  _obj->_ambient, _obj->_diffuse,
                                  _obj->_specular, _obj->_specular_exponent,
                                  _obj->_transmissive, _obj->_refractive_index,
                                  _obj->_ambient_edge, _obj->_diffuse_edge,
                                  _obj->_specular_edge, _obj->_specular_exponent_edge,
                                  _obj->_transmissive_edge, _obj->_refractive_index_edge,
                                  _obj->_scale,
                                  _obj->_edge_width, _obj->_edge_height,
                                  _bump_mapping_ptr == nullptr ? nullptr :_bump_mapping_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void BrickMaterial::clearGpuData()
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

void BrickMaterial::setAmbient(Light const &ambient)
{
    _obj->_ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuse(Light const &diffuse)
{
    _obj->_diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecular(Light const &specular)
{
    _obj->_specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->_specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissive(Light const &transmissive)
{
    _obj->_transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->_refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _obj->setBumpMapping(bm.get());
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void BrickMaterial::setAmbientEdge(Light const &ambient)
{
    _obj->_ambient_edge = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuseEdge(Light const &diffuse)
{
    _obj->_diffuse_edge = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularEdge(Light const &specular)
{
    _obj->_specular_edge = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExpEdge(int const &specular_exp)
{
    _obj->_specular_exponent_edge = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissiveEdge(Light const &transmissive)
{
    _obj->_transmissive_edge = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndexEdge(PREC const &ior)
{
    _obj->_refractive_index_edge = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setScale(PREC const &scale)
{
    _obj->_scale = scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeWidth(PREC const &edge_width)
{
    _obj->_edge_width = edge_width;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeHeight(PREC const &edge_height)
{
    _obj->_edge_height = edge_height;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
