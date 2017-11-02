#ifndef PX_CG_OBJECT_MATERIAL_TEXTURE_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_TEXTURE_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class Texture;
class BaseTexture;
class TextureMaterial;
class BaseTextureMaterial;
};

class px::Texture
{
public:
    enum class Format
    {
        RGB,
        RGBA
    };

    std::shared_ptr<Texture> static create(std::uint8_t * const &texture,
                                           int const &height,
                                           int const &width,
                                           Format const &format,
                                           PREC const &scale_u,
                                           PREC const &scale_v);
    std::shared_ptr<Texture> static create(std::string const &file,
                                           Format const &format,
                                           PREC const &scale_u,
                                           PREC const &scale_v);

    static std::uint8_t * loadTexture(std::string const &file,
                                      int &height, int &width,
                                      Format const &format);

    ~Texture();

    BaseTexture *obj() const;
    BaseTexture *devPtr();
    void up2Gpu();
    void clearGpuData();

    void setTexture(const std::uint8_t * const &texture,
                    int const &height,
                    int const &width,
                    Texture::Format const &format);
    void setScale(PREC const &scale_u, PREC const &scale_v);

protected:
    std::uint8_t * _texture;
    std::uint8_t * _texture_gpu;

    BaseTexture *_obj;
    BaseTexture *_dev_ptr;

    bool _need_upload;

    Texture(std::uint8_t * const &texture,
            int const &height,
            int const &width,
            Format const &format,
            PREC const &scale_u,
            PREC const &scale_v);
    Texture(std::string const &file,
            Format const &format,
            PREC const &scale_u,
            PREC const &scale_v);

    friend class TextureMaterial;
};



class px::BaseTexture
{
public:
    void setTexture(const std::uint8_t * const &texture,
                    int const &height,
                    int const &width,
                    Texture::Format const &format);
    void setScale(PREC const &scale_u, PREC const &scale_v);

    PX_CUDA_CALLABLE
    Light rgb(PREC const &u, PREC const &v) const;
    PX_CUDA_CALLABLE
    PREC alpha(PREC const &u, PREC const &v) const;

    ~BaseTexture() = default;

protected:

    BaseTexture(std::uint8_t * const &texture,
                int const &height,
                int const &width,
                Texture::Format const &format,
                PREC const &scale_u,
                PREC const &scale_v);

    Texture::Format _format;
    PREC _scale_u;
    PREC _scale_v;
    int _height;
    int _width;
    int _comp;

    std::uint8_t * _texture;

    friend class Texture;
};

class px::BaseTextureMaterial
{
public:
    PX_CUDA_CALLABLE
    static int getSpecularExp(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
    PX_CUDA_CALLABLE
    static PREC getRefractiveIndex(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
    PX_CUDA_CALLABLE
    static Light getAmbient(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
    PX_CUDA_CALLABLE
    static Light getDiffuse(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
    PX_CUDA_CALLABLE
    static Light getSpecular(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
    PX_CUDA_CALLABLE
    static Light getTransmissive(void * const &obj, PREC const &u, PREC const &v, PREC const &w);

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);
    void setTexture(const Texture * const &texture);

protected:
    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    PREC _refractive_index;

    const BaseTexture *_texture;

    BaseTextureMaterial(Light const &ambient,
                        Light const &diffuse,
                        Light const &specular,
                        int const &specular_exponent,
                        Light const &transmissive,
                        PREC const &refractive_index,
                        const Texture * const &texture);

    ~BaseTextureMaterial() = default;

    BaseTextureMaterial &operator=(BaseTextureMaterial const &) = delete;
    BaseTextureMaterial &operator=(BaseTextureMaterial &&) = delete;

    friend class TextureMaterial;
};


class px::TextureMaterial : public BaseMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient,
                                            Light const &diffuse,
                                            Light const &specular,
                                            int const &specular_exponent,
                                            Light const &transmissive,
                                            PREC const &refractive_index,
                                            std::shared_ptr<Texture> const & texture);

    void up2Gpu() override;
    void clearGpuData() override;

    int specularExp(PREC const &u, PREC const &v, PREC const &w) const override;
    PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const override;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);
    void setTexture(std::shared_ptr<Texture> const &texture);

    ~TextureMaterial();

protected:
    std::shared_ptr<Texture> _texture;

    BaseTextureMaterial *_obj;
    void *_gpu_obj;
    bool _need_upload;

    Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const override;

    TextureMaterial(Light const &ambient,
                    Light const &diffuse,
                    Light const &specular,
                    int const &specular_exponent,
                    Light const &transmissive,
                    PREC const &refractive_index,
                    std::shared_ptr<Texture> const & texture);

    TextureMaterial &operator=(TextureMaterial const &) = delete;
    TextureMaterial &operator=(TextureMaterial &&) = delete;
};

#endif // PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP