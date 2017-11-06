#ifndef PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class BaseUniformMaterial;
class UniformMaterial;
}

class px::BaseUniformMaterial
{
public:

    PX_CUDA_CALLABLE
    static PREC getShininess(void * const &obj, PREC const &u, PREC const &v, PREC const &w);
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
    void setShininess(PREC const &shininess);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);

protected:
    Light _ambient;
    Light _diffuse;
    Light _specular;
    PREC _shininessonent;
    Light _transmissive;
    PREC _refractive_index;

    BaseUniformMaterial(Light const &ambient,
                      Light const &diffuse,
                      Light const &specular,
                      PREC const &shininessonent,
                      Light const &transmissive,
                      PREC const &refractive_index);

    ~BaseUniformMaterial() = default;

    BaseUniformMaterial &operator=(BaseUniformMaterial const &) = delete;
    BaseUniformMaterial &operator=(BaseUniformMaterial &&) = delete;

    friend class UniformMaterial;
};


class px::UniformMaterial : public BaseMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient = {0, 0, 0},
                                                Light const &diffuse = {1, 1, 1},
                                                Light const &specular = {0, 0, 0},
                                                PREC const &shininessonent = 5,
                                                Light const &transmissive ={0, 0, 0},
                                                PREC const &refractive_index = 1.0);
    void up2Gpu() override;
    void clearGpuData() override ;

    PREC Shininess(PREC const &u, PREC const &v, PREC const &w) const override;
    PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const override;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setShininess(PREC const &shininess);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(PREC const &ior);

    ~UniformMaterial();

protected:
    BaseUniformMaterial *_obj;
    void *_gpu_obj;
    bool _need_upload;

    Light getAmbient(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getDiffuse(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getSpecular(PREC const &u, PREC const &v, PREC const &w) const override;
    Light getTransmissive(PREC const &u, PREC const &v, PREC const &w) const override;

    UniformMaterial(Light const &ambient,
                         Light const &diffuse,
                         Light const &specular,
                         PREC const &shininessonent,
                         Light const &transmissive,
                         PREC const &refractive_index);

    UniformMaterial &operator=(UniformMaterial const &) = delete;
    UniformMaterial &operator=(UniformMaterial &&) = delete;
};

#endif // PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP
