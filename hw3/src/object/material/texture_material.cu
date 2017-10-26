#include "object/material/texture_material.hpp"

#include "util/stb_image.h"
#include "util/stb_image_write.h"
#include <cstring>

#ifdef USE_CUDA
#include "gpu_creator.hpp"
#endif

using namespace px;

std::shared_ptr<Texture> Texture::create(std::uint8_t * const &texture,
                                         int const &height,
                                         int const &width,
                                         Format const &format,
                                         PREC const &scale_u,
                                         PREC const &scale_v)
{
    return std::shared_ptr<Texture>(new Texture(texture, height, width, format, scale_u, scale_v));
}
std::shared_ptr<Texture> Texture::create(std::string const &file,
                                         Format const &format,
                                         PREC const &scale_u,
                                         PREC const &scale_v)
{
    return std::shared_ptr<Texture>(new Texture(file, format, scale_u, scale_v));
}

Texture::Texture(std::string const &file,
                 Format const &format,
                 PREC const &scale_u,
                 PREC const &scale_v)
        : _format(format), _scale_u(scale_u), _scale_v(scale_v),
          _texture(nullptr), _dev_ptr(nullptr), _texture_gpu(nullptr),
          _need_upload(true), _gpu_data(false)
{
    _texture = loadTexture(file, _height, _width, format);
    _comp = format == Format::RGB ? 3 : 4;
}

Texture::Texture(std::uint8_t * const &texture,
                 int const &height,
                 int const &width,
                 Format const &format,
                 PREC const &scale_u,
                 PREC const &scale_v)
        : _format(format), _scale_u(scale_u), _scale_v(scale_v),
          _height(height), _width(width), _comp(format == Format::RGB ? 3 : 4),
          _texture(texture), _dev_ptr(nullptr), _texture_gpu(nullptr),
          _need_upload(true), _gpu_data(false)
{}

std::uint8_t * Texture::loadTexture(std::string const &file,
                                                int &height,
                                                int &width,
                                                Format const &format)
{
    int num_comp;
    auto data = stbi_load(file.data(), &width, &height, &num_comp, format == Format::RGB ? 3 : 4);

    if (data == nullptr)
        throw std::invalid_argument("[Error] Failed to load texture file `" + file + "`");

    return data;
}

void Texture::setTexture(const std::uint8_t * const &texture,
                         int const &height,
                         int const &width,
                         Format const &format)
{
    _height = height;
    _width = width;
    _format = format;
    _comp = format == Format::RGB ? 3 : 4;

    delete [] _texture;
    std::memcpy(_texture, texture, sizeof(std::uint8_t)*_height*_width*_comp);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}


void Texture::setScale(PREC const &scale_u, PREC const &scale_v)
{
    _scale_u = std::abs(scale_u), _scale_v = std::abs(scale_v);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}

PX_CUDA_CALLABLE
Light Texture::rgb(PREC const &u, PREC const &v) const
{
    // TODO bilinear interploration

    auto uu = std::abs(u/_scale_u);
    auto uv = std::abs(v/_scale_v);

    auto iu = static_cast<int>(uu) / _width;
    auto iv = static_cast<int>(uv) / _height;

    auto dsu = std::fmod(uu, static_cast<decltype(u)>(_width));
    auto dsv = std::fmod(uv, static_cast<decltype(u)>(_height));

    auto su = static_cast<int>(iu % 2 == 1 ? _width - dsu : dsu);
    auto sv = static_cast<int>(iv % 2 == 1 ? _height - dsv : dsv);

    auto tar = (sv * _width + su) * _comp;

    return {_texture[tar] /  PREC(255.0),
            _texture[tar+1] / PREC(255.0),
            _texture[tar+2] / PREC(255.0)};
}


PX_CUDA_CALLABLE
PREC Texture::alpha(PREC const &u, PREC const &v) const
{
    // TODO bilinear interploration

    if (_format == Format::RGB)
        return 1.0;

    auto uu = std::abs(u/_scale_u);
    auto uv = std::abs(v/_scale_v);

    auto iu = static_cast<int>(uu) / _width;
    auto iv = static_cast<int>(uv) / _height;

    auto dsu = std::fmod(uu, static_cast<PREC>(_width));
    auto dsv = std::fmod(uv, static_cast<PREC>(_height));

    auto su = static_cast<int>(iu % 2 == 1 ? _width - dsu : dsu);
    auto sv = static_cast<int>(iv % 2 == 1 ? _height - dsv : dsv);

    auto tar = (sv * _width + su) * _comp;

    return 1.0 - _texture[tar+3] / 255.0;
}

Texture *Texture::devPtr()
{
    return _dev_ptr;
}

void Texture::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(Texture)));

        if (_texture_gpu != nullptr)
            PX_CUDA_CHECK(cudaFree(_texture_gpu));

        PX_CUDA_CHECK(cudaMalloc(&_texture_gpu,
                                 sizeof(std::uint8_t)*_height*_width));
        PX_CUDA_CHECK(cudaMemcpy(_texture_gpu, _texture,
                                 sizeof(std::uint8_t)*_height*_width,
                                 cudaMemcpyHostToDevice));

        std::swap(_texture_gpu, _texture);
        _gpu_data = true;

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, this, sizeof(Texture),
                                 cudaMemcpyHostToDevice));

        std::swap(_texture_gpu, _texture);
        _gpu_data = false;

        _need_upload = false;
    }
#endif
}

void Texture::clearGpuData()
{
#ifdef USE_CUDA
    if (_texture_gpu != nullptr)
        PX_CUDA_CHECK(cudaFree(_texture_gpu));

    if (_dev_ptr == nullptr)
        return;

    PX_CUDA_CHECK(cudaFree(_dev_ptr));

    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

Texture::~Texture()
{
#ifdef USE_CUDA
    clearGpuData();
    if (_gpu_data == false)
#endif
    delete [] _texture;
}

BaseTextureMaterial::BaseTextureMaterial(Light const &ambient,
                                 Light const &diffuse,
                                 Light const &specular,
                                 const int &specular_exponent,
                                 Light const &transmissive,
                                 PREC const &refractive_index,
                                 const Texture * const &texture,
                                 const BumpMapping * const &bump_mapping)
        : BaseMaterial(bump_mapping),
          _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index),
          _texture(texture)
{}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    return _ambient * _texture->rgb(u, v);
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    return _diffuse * _texture->rgb(u, v);;
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getSpecular(PREC const &, PREC const &, PREC const &) const
{
    return _specular;
}

PX_CUDA_CALLABLE
int BaseTextureMaterial::specularExp(PREC const &, PREC const &, PREC const &) const
{
    return _specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    return _transmissive * (1-_texture->alpha(u, v));
}

PX_CUDA_CALLABLE
PREC BaseTextureMaterial::refractiveIndex(PREC const &, PREC const &, PREC const &) const
{
    return _refractive_index;
}


std::shared_ptr<Material> TextureMaterial::create(Light const &ambient,
                                                      Light const &diffuse,
                                                      Light const &specular,
                                                      const int &specular_exponent,
                                                      Light const &transmissive,
                                                      PREC const &refractive_index,
                                                      std::shared_ptr<Texture> const &texture,
                                                      std::shared_ptr<BumpMapping> const &bump_mapping)
{
    return std::shared_ptr<Material>(new TextureMaterial(ambient,
                                                             diffuse,
                                                             specular,
                                                             specular_exponent,
                                                             transmissive,
                                                             refractive_index,
                                                             texture,
                                                             bump_mapping));
}

TextureMaterial::TextureMaterial(Light const &ambient,
                                 Light const &diffuse,
                                 Light const &specular,
                                 const int &specular_exponent,
                                 Light const &transmissive,
                                 PREC const &refractive_index,
                                 std::shared_ptr<Texture> const &texture,
                                 std::shared_ptr<BumpMapping> const &bump_mapping)
        : _obj(new BaseTextureMaterial(ambient, diffuse,
                              specular, specular_exponent,
                              transmissive, refractive_index,
                              texture.get(), bump_mapping.get())),
          _base_obj(_obj),
          _texture_ptr(texture),
          _bump_mapping_ptr(bump_mapping),
          _dev_ptr(nullptr),
          _need_upload(true)
{}

TextureMaterial::~TextureMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}
BaseMaterial *const &TextureMaterial::obj() const noexcept
{
    return _base_obj;
}

BaseMaterial **TextureMaterial::devPtr()
{
    return _dev_ptr;
}
void TextureMaterial::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload || (_texture_ptr != nullptr && _texture_ptr->_need_upload))
    {
        clearGpuData();
        PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseMaterial**)));

        if (_bump_mapping_ptr != nullptr)
            _bump_mapping_ptr->up2Gpu();
        if (_texture_ptr != nullptr)
            _texture_ptr->up2Gpu();

        cudaDeviceSynchronize();

        GpuCreator::TextureMaterial(_dev_ptr,
                                    _obj->_ambient, _obj->_diffuse,
                                    _obj->_specular, _obj->_specular_exponent,
                                    _obj->_transmissive, _obj->_refractive_index,
                                    _texture_ptr == nullptr ? nullptr : _texture_ptr->devPtr(),
                                    _bump_mapping_ptr == nullptr ? nullptr : _bump_mapping_ptr->devPtr());

        _need_upload = false;
    }
#endif
}

void TextureMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_dev_ptr == nullptr)
        return;

    if (_bump_mapping_ptr.use_count() == 1)
        _bump_mapping_ptr->clearGpuData();

    if (_texture_ptr.use_count() == 1)
        _texture_ptr->clearGpuData();

    GpuCreator::destroy(_dev_ptr);

    _dev_ptr = nullptr;
    _need_upload = true;
#endif
}

void TextureMaterial::setAmbient(Light const &ambient)
{
    _obj->_ambient = ambient;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setDiffuse(Light const &diffuse)
{
    _obj->_diffuse = diffuse;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setSpecular(Light const &specular)
{
    _obj->_specular = specular;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->_specular_exponent = specular_exp;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setTransmissive(Light const &transmissive)
{
    _obj->_transmissive = transmissive;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->_refractive_index = ior;
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setBumpMapping(std::shared_ptr<BumpMapping> const &bm)
{
    _bump_mapping_ptr = bm;
    _obj->setBumpMapping(bm.get());
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setTexture(std::shared_ptr<Texture> const &tt)
{
    _texture_ptr = tt;
    _obj->_texture = tt.get();
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
