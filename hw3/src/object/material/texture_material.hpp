#ifndef PX_CG_OBJECT_MATERIAL_TEXTURE_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_TEXTURE_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class Texture;
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

    void setTexture(const std::uint8_t * const &texture,
                    int const &height,
                    int const &width,
                    Format const &format);
    void setScale(PREC const &scale_u, PREC const &scale_v);

    PX_CUDA_CALLABLE
    Light rgb(PREC const &u, PREC const &v) const;
    PX_CUDA_CALLABLE
    PREC alpha(PREC const &u, PREC const &v) const;

    ~Texture();

    Texture *devPtr();
    void up2Gpu();
    void clearGpuData();

protected:
    Format _format;
    PREC _scale_u;
    PREC _scale_v;
    int _height;
    int _width;
    int _comp;

    std::uint8_t * _texture;

    Texture * _dev_ptr;
    std::uint8_t * _texture_gpu;

    bool _need_upload;
    bool _gpu_data;

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


class px::BaseTextureMaterial : public BaseMaterial
{
public:
    PX_CUDA_CALLABLE
    BaseTextureMaterial(Light const &ambient,
                        Light const &diffuse,
                        Light const &specular,
                        int const &specular_exponent,
                        Light const &transmissive, // final transmissive = (1.0 - alpha/255.0) + transmissive
                        PREC const &refractive_index,
                        const Texture *const &texture,
                        const BumpMapping *const &bump_mapping);
    PX_CUDA_CALLABLE
    ~BaseTextureMaterial() = default;

    PX_CUDA_CALLABLE
    int specularExp(PREC const &u, PREC const &v, PREC const &w) const override;
    PX_CUDA_CALLABLE
    PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const override;

protected:
    PX_CUDA_CALLABLE
    Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const override;
    PX_CUDA_CALLABLE
    Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const override;
    PX_CUDA_CALLABLE
    Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const override;
    PX_CUDA_CALLABLE
    Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const override;

    BaseTextureMaterial &operator=(BaseTextureMaterial const &) = delete;
    BaseTextureMaterial &operator=(BaseTextureMaterial &&) = delete;

    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    PREC _refractive_index;

    const Texture *_texture;

    friend class TextureMaterial;
};



class px::TextureMaterial : public Material
{
public:

    std::shared_ptr<Material> static create(Light const &ambient,   // final ambient = ambient * RGB/255.0
                                                Light const &diffuse,   // final ambient = diffuse * RGB/255.0
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive, // final transmissive = (1.0 - alpha/255.0) * transmissive
                                                PREC const &refractive_index,
                                                std::shared_ptr<Texture> const & texture,
                                                std::shared_ptr<BumpMapping> const &bump_mapping=nullptr);
    BaseMaterial *const &obj() const noexcept override;
    BaseMaterial **devPtr() override;
    void up2Gpu() override;
    void clearGpuData() override ;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);
    void setBumpMapping(std::shared_ptr<BumpMapping> const &bump_mapping);
    void setTexture(std::shared_ptr<Texture> const &texture);

    ~TextureMaterial();
protected:
    BaseTextureMaterial *_obj;
    BaseMaterial *_base_obj;

    TextureMaterial(Light const &ambient,
                    Light const &diffuse,
                    Light const &specular,
                    int const &specular_exponent,
                    Light const &transmissive,
                    PREC const &refractive_index,
                    std::shared_ptr<Texture> const & texture,
                    std::shared_ptr<BumpMapping> const &bump_mapping);

    TextureMaterial &operator=(TextureMaterial const &) = delete;
    TextureMaterial &operator=(TextureMaterial &&) = delete;

    std::shared_ptr<Texture> _texture_ptr;
    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial **_dev_ptr;
    bool _need_upload;
};

#endif // PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP