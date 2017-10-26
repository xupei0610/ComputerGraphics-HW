#ifndef PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class UniformMaterial;
class BaseUniformMaterial;
}

class px::BaseUniformMaterial : public BaseMaterial
{
public:
    PX_CUDA_CALLABLE
    BaseUniformMaterial(Light const &ambient,
                        Light const &diffuse,
                        Light const &specular,
                        int const &specular_exponent,
                        Light const &transmissive,
                        PREC const &refractive_index,
                        const BumpMapping * const &bump_mapping);
    PX_CUDA_CALLABLE
    ~BaseUniformMaterial() = default;

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


    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    PREC _refractive_index;

    BaseUniformMaterial &operator=(BaseUniformMaterial const &) = delete;
    BaseUniformMaterial &operator=(BaseUniformMaterial &&) = delete;

    friend class UniformMaterial;
};


class px::UniformMaterial : public Material
{
public:

    static std::shared_ptr<Material> create(Light const &ambient = {0, 0, 0},
                                                Light const &diffuse = {1, 1, 1},
                                                Light const &specular = {0, 0, 0},
                                                int const &specular_exponent = 5,
                                                Light const &transmissive ={0, 0, 0},
                                                PREC const &refractive_index = 1.0,
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

    ~UniformMaterial();
protected:
    BaseUniformMaterial *_obj;
    BaseMaterial *_base_obj;

    UniformMaterial(Light const &ambient,
                    Light const &diffuse,
                    Light const &specular,
                    int const &specular_exponent,
                    Light const &transmissive,
                    PREC const &refractive_index,
                    std::shared_ptr<BumpMapping> const &bm);

    UniformMaterial &operator=(UniformMaterial const &) = delete;
    UniformMaterial &operator=(UniformMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial **_dev_ptr;
    bool _need_upload;

};

#endif // PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP
