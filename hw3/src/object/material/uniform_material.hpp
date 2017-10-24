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
    ~BaseUniformMaterial() = default;

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

    BaseUniformMaterial(Light const &ambient,
                        Light const &diffuse,
                        Light const &specular,
                        int const &specular_exponent,
                        Light const &transmissive,
                        double const &refractive_index,
                        const BumpMapping * const &bump_mapping);

    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    double _refractive_index;

    BaseUniformMaterial &operator=(BaseUniformMaterial const &) = delete;
    BaseUniformMaterial &operator=(BaseUniformMaterial &&) = delete;
};


class px::UniformMaterial : public BaseUniformMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient = {0, 0, 0},
                                                Light const &diffuse = {1, 1, 1},
                                                Light const &specular = {0, 0, 0},
                                                int const &specular_exponent = 5,
                                                Light const &transmissive ={0, 0, 0},
                                                double const &refractive_index = 1.0,
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

    ~UniformMaterial();
protected:
    UniformMaterial(Light const &ambient,
                    Light const &diffuse,
                    Light const &specular,
                    int const &specular_exponent,
                    Light const &transmissive,
                    double const &refractive_index,
                    std::shared_ptr<BumpMapping> const &bm);

    UniformMaterial &operator=(UniformMaterial const &) = delete;
    UniformMaterial &operator=(UniformMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial *_dev_ptr;
    bool _need_upload;

};

#endif // PX_CG_OBJECT_MATERIAL_UNIFORM_MATERIAL_HPP
