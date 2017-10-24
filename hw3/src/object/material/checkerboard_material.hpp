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

    ~BaseCheckerboardMaterial() = default;

    PX_CUDA_CALLABLE
    int specularExp(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    double refractiveIndex(double const &u, double const &v, double const &w) const override;

protected:
    PX_CUDA_CALLABLE
    Light getAmbient(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getDiffuse(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getSpecular(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getTransmissive(double const &u, double const &v, double const &w) const override;

    BaseCheckerboardMaterial(Light const &ambient,
                             Light const &diffuse,
                             Light const &specular,
                             int const &specular_exponent,
                             Light const &transmissive,
                             double const &refractive_index,
                             double const &dim_scale,
                             double const &color_scale,
                             const BumpMapping *const &bump_mapping);
    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    double _refractive_index;
    double _dim_scale;
    double _color_scale;
};


class px::CheckerboardMaterial : public BaseCheckerboardMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient,
                                                Light const &diffuse,
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive,
                                                double const &refractive_index,
                                                double const &dim_scale,
                                                double const &color_scale,
                                                std::shared_ptr<BumpMapping> const &bump_mapping=nullptr);

    BaseMaterial *up2Gpu() override;
    void clearGpuData() override ;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(double const &ior);
    void setBumpMapping(std::shared_ptr<BumpMapping> const &bump_mapping);
    void setDimScale(double const &dim_scale);
    void setColorScale(double const &color_scale);

    ~CheckerboardMaterial();
protected:
    CheckerboardMaterial(Light const &ambient,
                         Light const &diffuse,
                         Light const &specular,
                         int const &specular_exponent,
                         Light const &transmissive,
                         double const &refractive_index,
                         double const &dim_scale,
                         double const &color_scale,
                         std::shared_ptr<BumpMapping> const &bump_mapping);

    CheckerboardMaterial &operator=(CheckerboardMaterial const &) = delete;
    CheckerboardMaterial &operator=(CheckerboardMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial *_dev_ptr;
    bool _need_upload;
};

#endif // PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP