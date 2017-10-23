#include "object/material/brick_material.hpp"

using namespace px;

BaseBrickMaterial::BaseBrickMaterial(Light const &ambient,
                                     Light const &diffuse,
                                     Light const &specular,
                                     int const &specular_exponent,
                                     Light const &transmissive,
                                     double const &refractive_index,
                                     Light const &ambient_edge,
                                     Light const &diffuse_edge,
                                     Light const &specular_edge,
                                     int const &specular_exponent_edge,
                                     Light const &transmissive_edge,
                                     double const &refractive_index_edge,
                                     double const &scale,
                                     double const &edge_width,
                                     double const &edge_height,
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
Light BaseBrickMaterial::getAmbient(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _ambient_edge;
    return _ambient;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getDiffuse(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _diffuse_edge;
    return _diffuse;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getSpecular(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _specular_edge;
    return _specular;
}

PX_CUDA_CALLABLE
int BaseBrickMaterial::specularExp(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _specular_exponent_edge;
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseBrickMaterial::getTransmissive(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _transmissive_edge;
    return _transmissive;
}

PX_CUDA_CALLABLE
double BaseBrickMaterial::refractiveIndex(double const &u, double const &v, double const &w) const
{
    if (onEdge(u, v, w))
        return _refractive_index_edge;
    return _refractive_index;
}

bool BaseBrickMaterial::onEdge(double const &u,
                               double const &v,
                               double const &w) const noexcept
{
    auto tx = static_cast<int>(u < 0 ? std::ceil(_scale*u) : std::floor(_scale*u));
    auto ty = static_cast<int>(v < 0 ? std::ceil(_scale*v) : std::floor(_scale*v));
    return ((std::abs(_scale*u - tx) < _edge_width) && ((tx & 0x0001) == (ty & 0x0001))) || (std::abs(_scale*v - ty) < _edge_height);
}

std::shared_ptr<BaseMaterial> BrickMaterial::create(Light const &ambient,
                                                    Light const &diffuse,
                                                    Light const &specular,
                                                    int const &specular_exponent,
                                                    Light const &transmissive,
                                                    double const &refractive_index,
                                                    Light const &ambient_edge,
                                                    Light const &diffuse_edge,
                                                    Light const &specular_edge,
                                                    int const &specular_exponent_edge,
                                                    Light const &transmissive_edge,
                                                    double const &refractive_index_edge,
                                                    double const &scale,
                                                    double const &edge_width,
                                                    double const &edge_height,
                                                    std::shared_ptr<BumpMapping> const &bump_mapping)
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
                                                           edge_height,
                                                           bump_mapping));
}

BrickMaterial::BrickMaterial(Light const &ambient,
                             Light const &diffuse,
                             Light const &specular,
                             int const &specular_exponent,
                             Light const &transmissive,
                             double const &refractive_index,
                             Light const &ambient_edge,
                             Light const &diffuse_edge,
                             Light const &specular_edge,
                             int const &specular_exponent_edge,
                             Light const &transmissive_edge,
                             double const &refractive_index_edge,
                             double const &scale,
                             double const &edge_width,
                             double const &edge_height,
                             std::shared_ptr<BumpMapping> const &bump_mapping)
        : BaseBrickMaterial(ambient, diffuse,
                            specular, specular_exponent,
                            transmissive, refractive_index,
                            ambient_edge, diffuse_edge,
                            specular_edge, specular_exponent_edge,
                            transmissive_edge, refractive_index_edge,
                            scale, edge_width, edge_height,
                            bump_mapping.get()),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

BaseMaterial* BrickMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseBrickMaterial)));

        _bump_mapping = _bump_mapping_ptr->up2Gpu();

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr,
                                 dynamic_cast<BaseBrickMaterial*>(this),
                                 sizeof(BaseBrickMaterial),
                                 cudaMemcpyHostToDevice));

        _bump_mapping = _bump_mapping_ptr.get();
        _need_upload = false;
    }
    return _dev_ptr;
#else
    return this;
#endif
}

void BrickMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr))
    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void BrickMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndex(double const &ior)
{
    _refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _bump_mapping = bm.get();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void BrickMaterial::setAmbientEdge(Light const &ambient)
{
    _ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setDiffuseEdge(Light const &diffuse)
{
    _diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularEdge(Light const &specular)
{
    _specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setSpecularExpEdge(int const &specular_exp)
{
    _specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setTransmissiveEdge(Light const &transmissive)
{
    _transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setRefractiveIndexEdge(double const &ior)
{
    _refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setScale(double const &scale)
{
    _scale = scale;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeWidth(double const &edge_width)
{
    _edge_width = edge_width;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void BrickMaterial::setEdgeHeight(double const &edge_height)
{
    _edge_height = edge_height;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
