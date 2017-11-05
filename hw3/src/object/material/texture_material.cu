#include "object/material/texture_material.hpp"

#include "util/stb_image.h"
#include "util/stb_image_write.h"
#include <cstring>

using namespace px;

BaseTexture::BaseTexture(std::uint8_t *const &texture, int const &height,
                         int const &width, Texture::Format const &format,
                         PREC const &scale_u, PREC const &scale_v)
        : _format(format), _scale_u(scale_u), _scale_v(scale_v),
          _height(height), _width(width), _comp(format == Texture::Format::RGB ? 3 : 4),
          _texture(texture)
{}

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
        : _texture(nullptr), _texture_gpu(nullptr), _dev_ptr(nullptr),
          _need_upload(true)
{
    int height, width;
    _texture = loadTexture(file, height, width, format);
    _obj = new BaseTexture(_texture, height, width, format, scale_u, scale_v);
}

Texture::Texture(std::uint8_t * const &texture,
                 int const &height,
                 int const &width,
                 Format const &format,
                 PREC const &scale_u,
                 PREC const &scale_v)
        : _texture(texture), _texture_gpu(nullptr),
          _obj(new BaseTexture(texture, height, width, format, scale_u, scale_v)), _dev_ptr(nullptr),
          _need_upload(true)
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

void BaseTexture::setTexture(const std::uint8_t * const &texture,
                         int const &height,
                         int const &width,
                         Texture::Format const &format)
{
    _height = height;
    _width = width;
    _format = format;
    _comp = format == Texture::Format::RGB ? 3 : 4;

    delete [] _texture;
    std::memcpy(_texture, texture, sizeof(std::uint8_t)*_height*_width*_comp);
}


void BaseTexture::setScale(PREC const &scale_u, PREC const &scale_v)
{
    _scale_u = std::abs(scale_u), _scale_v = std::abs(scale_v);
}

BaseTexture* Texture::obj() const
{
    return _obj;
}

PX_CUDA_CALLABLE
Light BaseTexture::rgb(PREC const &u, PREC const &v) const
{
    // TODO bilinear interploration

    auto uu = std::abs(u/_scale_u);;
    auto uv = std::abs(v/_scale_v);

    auto iu = static_cast<int>((static_cast<int>(uu) / _width) % 2 == 1 ? _width - std::fmod(uu, static_cast<decltype(u)>(_width)) : std::fmod(uu, static_cast<decltype(u)>(_width)));
    auto iv = static_cast<int>((static_cast<int>(uv) / _height) % 2 == 1 ? _height - std::fmod(uv, static_cast<decltype(u)>(_height)) : std::fmod(uv, static_cast<decltype(u)>(_height)));

    iu = (iv * _width + iu) * _comp;

    return {_texture[iu] /  PREC(255.0),
            _texture[iu+1] / PREC(255.0),
            _texture[iu+2] / PREC(255.0)};
}


PX_CUDA_CALLABLE
PREC BaseTexture::alpha(PREC const &u, PREC const &v) const
{
    // TODO bilinear interploration

    if (_comp == 3)
        return 1.0;

    auto uu = std::abs(u/_scale_u);;
    auto uv = std::abs(v/_scale_v);

    auto iu = static_cast<int>((static_cast<int>(uu) / _width) % 2 == 1 ?
                               _width - std::fmod(uu, static_cast<decltype(u)>(_width)) :
                               std::fmod(uu, static_cast<decltype(u)>(_width)));
    auto iv = static_cast<int>((static_cast<int>(uv) / _height) % 2 == 1 ?
                               _height - std::fmod(uv, static_cast<decltype(u)>(_height)) :
                               std::fmod(uv, static_cast<decltype(u)>(_height)));

    iu = (iv * _width + iu) * _comp;

    return 1.0 - _texture[iu+3] / PREC(255.0);

//    auto uu = std::abs(u/_scale_u);
//    auto uv = std::abs(v/_scale_v);
//
//    auto iu = static_cast<int>(uu) / _width;
//    auto iv = static_cast<int>(uv) / _height;
//
//    auto dsu = std::fmod(uu, static_cast<PREC>(_width));
//    auto dsv = std::fmod(uv, static_cast<PREC>(_height));
//
//    auto su = static_cast<int>(iu % 2 == 1 ? _width - dsu : dsu);
//    auto sv = static_cast<int>(iv % 2 == 1 ? _height - dsv : dsv);
//
//    auto tar = (sv * _width + su) * _comp;
//
//    return 1.0 - _texture[tar+3] / 255.0;
}

BaseTexture *Texture::devPtr()
{
    return _dev_ptr;
}

void Texture::up2Gpu()
{
#ifdef USE_CUDA
    if (_need_upload)
    {
        if (_dev_ptr == nullptr)
            PX_CUDA_CHECK(cudaMalloc(&_dev_ptr, sizeof(BaseTexture)));

        if (_texture_gpu != nullptr)
            PX_CUDA_CHECK(cudaFree(_texture_gpu));

        PX_CUDA_CHECK(cudaMalloc(&_texture_gpu,
                                 sizeof(std::uint8_t)*_obj->_comp*_obj->_height*_obj->_width));
        PX_CUDA_CHECK(cudaMemcpy(_texture_gpu, _texture,
                                 sizeof(std::uint8_t)*_obj->_comp*_obj->_height*_obj->_width,
                                 cudaMemcpyHostToDevice));

        _obj->_texture = _texture_gpu;

        PX_CUDA_CHECK(cudaMemcpy(_dev_ptr, _obj, sizeof(BaseTexture),
                                 cudaMemcpyHostToDevice));

        _obj->_texture = _texture;

        _need_upload = false;
    }
#endif
}

void Texture::clearGpuData()
{
#ifdef USE_CUDA
    if (_texture_gpu != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_texture_gpu));
        _texture_gpu = nullptr;
    }

    if (_dev_ptr != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_dev_ptr));
        _dev_ptr = nullptr;
    }

    _need_upload = true;
#endif
}

Texture::~Texture()
{
#ifdef USE_CUDA
    clearGpuData();
#endif
    delete [] _texture;
    delete _obj;
}

void Texture::setTexture(const std::uint8_t * const &texture,
                             int const &height,
                             int const &width,
                             Format const &format)
{

    _obj->setTexture(texture, height, width, format);

#ifdef USE_CUDA
    _need_upload = true;
#endif
}


void Texture::setScale(PREC const &scale_u, PREC const &scale_v)
{
    _obj->setScale(scale_u, scale_v);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}


BaseTextureMaterial::BaseTextureMaterial(Light const &ambient,
                                         Light const &diffuse,
                                         Light const &specular,
                                         int const &specular_exponent,
                                         Light const &transmissive,
                                         PREC const &refractive_index,
                                         const Texture *const & texture)
        : _ambient(ambient),
          _diffuse(diffuse),
          _specular(specular),
          _specular_exponent(specular_exponent),
          _transmissive(transmissive),
          _refractive_index(refractive_index),
          _texture(texture->obj())
{}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getAmbient(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseTextureMaterial*>(obj);
    return o->_ambient * o->_texture->rgb(u, v);
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getDiffuse(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseTextureMaterial*>(obj);
    return o->_diffuse * o->_texture->rgb(u, v);
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getSpecular(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseTextureMaterial*>(obj)->_specular;
}

PX_CUDA_CALLABLE
int BaseTextureMaterial::getSpecularExp(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseTextureMaterial*>(obj)->_specular_exponent;
}

PX_CUDA_CALLABLE
Light BaseTextureMaterial::getTransmissive(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    auto o = reinterpret_cast<BaseTextureMaterial*>(obj);
    return o->_transmissive * (1-o->_texture->alpha(u, v));;
}

PX_CUDA_CALLABLE
PREC BaseTextureMaterial::getRefractiveIndex(void *const &obj, PREC const &u, PREC const &v, PREC const &w)
{
    return reinterpret_cast<BaseTextureMaterial*>(obj)->_refractive_index;
}

void BaseTextureMaterial::setAmbient(Light const &ambient)
{
    _ambient = ambient;
}
void BaseTextureMaterial::setDiffuse(Light const &diffuse)
{
    _diffuse = diffuse;
}
void BaseTextureMaterial::setSpecular(Light const &specular)
{
    _specular = specular;
}
void BaseTextureMaterial::setSpecularExp(int const &specular_exp)
{
    _specular_exponent = specular_exp;
}
void BaseTextureMaterial::setTransmissive(Light const &transmissive)
{
    _transmissive = transmissive;
}
void BaseTextureMaterial::setRefractiveIndex(PREC const &ior)
{
    _refractive_index = ior;
}
void BaseTextureMaterial::setTexture(const Texture * const &texture)
{
    _texture = texture->obj();
}

std::shared_ptr<BaseMaterial> TextureMaterial::create(Light const &ambient,
                                                  Light const &diffuse,
                                                  Light const &specular,
                                                  int const &specular_exponent,
                                                  Light const &transmissive,
                                                  PREC const &refractive_index,
                                                  std::shared_ptr<Texture> const & texture)
{
    return std::shared_ptr<BaseMaterial>(new TextureMaterial(ambient,
                                                         diffuse,
                                                         specular,
                                                         specular_exponent,
                                                         transmissive,
                                                         refractive_index,
                                                         texture));
}

TextureMaterial::TextureMaterial(Light const &ambient,
                                 Light const &diffuse,
                                 Light const &specular,
                                 int const &specular_exponent,
                                 Light const &transmissive,
                                 PREC const &refractive_index,
                                 std::shared_ptr<Texture> const & texture)
        : BaseMaterial(),
          _texture(texture),
          _obj(new BaseTextureMaterial(ambient, diffuse,
                                       specular, specular_exponent,
                                       transmissive, refractive_index, texture.get())),

          _gpu_obj(nullptr),
          _need_upload(true)
{}

TextureMaterial::~TextureMaterial()
{
    delete _obj;
#ifdef USE_CUDA
    clearGpuData();
#endif
}

#ifdef USE_CUDA
__device__ fnAmbient_t __fn_ambient_texture_material = BaseTextureMaterial::getAmbient;
__device__ fnDiffuse_t __fn_diffuse_texture_material = BaseTextureMaterial::getDiffuse;
__device__ fnSpecular_t __fn_specular_texture_material = BaseTextureMaterial::getSpecular;
__device__ fnSpecularExp_t __fn_specular_exp_texture_material = BaseTextureMaterial::getSpecularExp;
__device__ fnTransmissive_t __fn_transmissive_texture_material = BaseTextureMaterial::getTransmissive;
__device__ fnRefractiveIndex_t __fn_refractive_index_texture_material = BaseTextureMaterial::getRefractiveIndex;
#endif

void TextureMaterial::up2Gpu()
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
            PX_CUDA_CHECK(cudaMalloc(&_gpu_obj, sizeof(BaseTextureMaterial)));
            PX_CUDA_CHECK(cudaMalloc(&dev_ptr, sizeof(MaterialObj)));
        }
        if (fn_ambient_h == nullptr)
        {
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_ambient_h, __fn_ambient_texture_material, sizeof(fnAmbient_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_diffuse_h, __fn_diffuse_texture_material, sizeof(fnDiffuse_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_h, __fn_specular_texture_material, sizeof(fnSpecular_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_specular_exp_h, __fn_specular_exp_texture_material, sizeof(fnSpecularExp_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_transmissive_h, __fn_transmissive_texture_material, sizeof(fnTransmissive_t)));
            PX_CUDA_CHECK(cudaMemcpyFromSymbol(&fn_refractive_index_h, __fn_refractive_index_texture_material, sizeof(fnRefractiveIndex_t)));
        }

        _texture->up2Gpu();

        _obj->_texture = _texture->devPtr();

        PX_CUDA_CHECK(cudaMemcpy(_gpu_obj, _obj, sizeof(BaseTextureMaterial),
                                 cudaMemcpyHostToDevice));
        MaterialObj tmp(_gpu_obj,
                        fn_ambient_h, fn_diffuse_h,
                        fn_specular_h, fn_specular_exp_h,
                        fn_transmissive_h, fn_refractive_index_h);

        PX_CUDA_CHECK(cudaMemcpy(dev_ptr, &tmp, sizeof(MaterialObj),
                                 cudaMemcpyHostToDevice));

        _obj->_texture = _texture->obj();
        _need_upload = false;
    }
#endif
}

void TextureMaterial::clearGpuData()
{
#ifdef USE_CUDA
    if (_gpu_obj != nullptr)
    {
        PX_CUDA_CHECK(cudaFree(_gpu_obj));
        _gpu_obj = nullptr;
    }
    if (_texture.use_count() == 1)
        _texture->clearGpuData();
    BaseMaterial::clearGpuData();
#endif
}

int TextureMaterial::specularExp(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getSpecularExp(_obj, u, v, w);
}
PREC TextureMaterial::refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getRefractiveIndex(_obj, u, v, w);
}
Light TextureMaterial::getAmbient(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getAmbient(_obj, u, v, w);
}
Light TextureMaterial::getDiffuse(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getDiffuse(_obj, u, v, w);
}
Light TextureMaterial::getSpecular(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getSpecular(_obj, u, v, w);
}
Light TextureMaterial::getTransmissive(PREC const &u, PREC const &v, PREC const &w) const
{
    return BaseTextureMaterial::getTransmissive(_obj, u, v, w);
}

void TextureMaterial::setAmbient(Light const &ambient)
{
    _obj->setAmbient(ambient);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setDiffuse(Light const &diffuse)
{
    _obj->setDiffuse(diffuse);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setSpecular(Light const &specular)
{
    _obj->setSpecular(specular);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setSpecularExp(int const &specular_exp)
{
    _obj->setSpecularExp(specular_exp);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setTransmissive(Light const &transmissive)
{
    _obj->setTransmissive(transmissive);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
void TextureMaterial::setRefractiveIndex(PREC const &ior)
{
    _obj->setRefractiveIndex(ior);
#ifdef USE_CUDA
    _need_upload = true;
#endif
}

void TextureMaterial::setTexture(std::shared_ptr<Texture> const &texture)
{
    _texture = texture;
    _obj->setTexture(texture.get());
#ifdef USE_CUDA
    _need_upload = true;
#endif
}
