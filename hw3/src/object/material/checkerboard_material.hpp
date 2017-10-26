#ifndef PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class BaseCheckerboardMaterial;
class CheckerboardMaterial;
}

class px::BaseCheckerboardMaterial : public BaseMaterial
{
public:
    PX_CUDA_CALLABLE
    BaseCheckerboardMaterial(Light const &ambient,
                             Light const &diffuse,
                             Light const &specular,
                             int const &specular_exponent,
                             Light const &transmissive,
                             PREC const &refractive_index,
                             PREC const &dim_scale,
                             PREC const &color_scale,
                             const BumpMapping *const &bump_mapping);
    PX_CUDA_CALLABLE
    ~BaseCheckerboardMaterial() = default;

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
    PREC _dim_scale;
    PREC _color_scale;

    friend class CheckerboardMaterial;
};


class px::CheckerboardMaterial : public Material
{
public:

    static std::shared_ptr<Material> create(Light const &ambient,
                                                Light const &diffuse,
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive,
                                                PREC const &refractive_index,
                                                PREC const &dim_scale,
                                                PREC const &color_scale,
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
    void setDimScale(PREC const &dim_scale);
    void setColorScale(PREC const &color_scale);

    ~CheckerboardMaterial();
protected:
    BaseCheckerboardMaterial *_obj;
    BaseMaterial *_base_obj;

    CheckerboardMaterial(Light const &ambient,
                         Light const &diffuse,
                         Light const &specular,
                         int const &specular_exponent,
                         Light const &transmissive,
                         PREC const &refractive_index,
                         PREC const &dim_scale,
                         PREC const &color_scale,
                         std::shared_ptr<BumpMapping> const &bump_mapping);

    CheckerboardMaterial &operator=(CheckerboardMaterial const &) = delete;
    CheckerboardMaterial &operator=(CheckerboardMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial **_dev_ptr;
    bool _need_upload;
};

#endif // PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP