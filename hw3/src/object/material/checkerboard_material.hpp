#ifndef PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class BaseCheckerboardMaterial;
class CheckerboardMaterial;
}

class px::BaseCheckerboardMaterial
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
    void setDimScale(PREC const &s);
    void setColorScale(PREC const &s);

protected:
    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    PREC _refractive_index;

    PREC _dim_scale;
    PREC _color_scale;

    BaseCheckerboardMaterial(Light const &ambient,
                      Light const &diffuse,
                      Light const &specular,
                      int const &specular_exponent,
                      Light const &transmissive,
                      PREC const &refractive_index,
                      PREC const &dim_scale,
                      PREC const &color_scale);

    ~BaseCheckerboardMaterial() = default;

    BaseCheckerboardMaterial &operator=(BaseCheckerboardMaterial const &) = delete;
    BaseCheckerboardMaterial &operator=(BaseCheckerboardMaterial &&) = delete;

    friend class CheckerboardMaterial;
};


class px::CheckerboardMaterial : public BaseMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient,
                                            Light const &diffuse,
                                            Light const &specular,
                                            int const &specular_exponent,
                                            Light const &transmissive,
                                            PREC const &refractive_index,
                                            PREC const &dim_scale,
                                            PREC const &color_scale);
    void up2Gpu() override;
    void clearGpuData() override ;

    int specularExp(PREC const &u, PREC const &v, PREC const &w) const override;
    PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const override;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);
    void setDimScale(PREC const &s);
    void setColorScale(PREC const &s);

    ~CheckerboardMaterial();

protected:
    BaseCheckerboardMaterial *_obj;
    void *_gpu_obj;
    bool _need_upload;

    Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const override;

    CheckerboardMaterial(Light const &ambient,
                         Light const &diffuse,
                         Light const &specular,
                         int const &specular_exponent,
                         Light const &transmissive,
                         PREC const &refractive_index,
                         PREC const &dim_scale,
                         PREC const &color_scale);

    CheckerboardMaterial &operator=(CheckerboardMaterial const &) = delete;
    CheckerboardMaterial &operator=(CheckerboardMaterial &&) = delete;
};

#endif // PX_CG_OBJECT_MATERIAL_CHECKERBOARD_MATERIAL_HPP
