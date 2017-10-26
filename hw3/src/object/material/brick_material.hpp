#ifndef PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP
#define PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP

#include "object/material/base_material.hpp"

namespace px
{
class BaseBrickMaterial;
class BrickMaterial;
}

class px::BaseBrickMaterial : public BaseMaterial
{
public:
    PX_CUDA_CALLABLE
    BaseBrickMaterial(Light const &ambient,
                      Light const &diffuse,
                      Light const &specular,
                      int const &specular_exponent,
                      Light const &transmissive,
                      PREC const &refractive_index,
                      Light const &ambient_edge,
                      Light const &diffuse_edge,
                      Light const &specular_edge,
                      int const &specular_exponent_edge,
                      Light const &transmissive_edge,
                      PREC const &refractive_index_edge,
                      PREC const &scale,
                      PREC const &edge_width,
                      PREC const &edge_height,
                      const BumpMapping *const &bump_mapping);
    PX_CUDA_CALLABLE
    ~BaseBrickMaterial() = default;

    PX_CUDA_CALLABLE
    int specularExp(PREC const &u, PREC const &v, PREC const &w) const override;
    PX_CUDA_CALLABLE
    PREC refractiveIndex(PREC const &u, PREC const &v, PREC const &w) const override;

    PX_CUDA_CALLABLE
    bool onEdge(PREC const &u, PREC const &v, PREC const &w) const noexcept;

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

    Light _ambient_edge;
    Light _diffuse_edge;
    Light _specular_edge;
    int _specular_exponent_edge;
    Light _transmissive_edge;
    PREC _refractive_index_edge;

    PREC _scale;
    PREC _edge_width;
    PREC _edge_height;

    friend class BrickMaterial;
};


class px::BrickMaterial : public Material
{
public:

    static std::shared_ptr<Material> create(Light const &ambient,
                                                Light const &diffuse,
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive,
                                                PREC const &refractive_index,
                                                Light const &ambient_edge,
                                                Light const &diffuse_edge,
                                                Light const &specular_edge,
                                                int const &specular_exponent_edge,
                                                Light const &transmissive_edge,
                                                PREC const &refractive_index_edge,
                                                PREC const &scale,
                                                PREC const &edge_width,
                                                PREC const &edge_height,
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
    void setAmbientEdge(Light const &ambient);
    void setDiffuseEdge(Light const &diffuse);
    void setSpecularEdge(Light const &specular);
    void setSpecularExpEdge(int const &specular_exp);
    void setTransmissiveEdge(Light const &transmissive);
    void setRefractiveIndexEdge(PREC const &ior);
    void setBumpMapping(std::shared_ptr<BumpMapping> const &bump_mapping);
    void setScale(PREC const &scale);
    void setEdgeWidth(PREC const &width);
    void setEdgeHeight(PREC const &height);

    ~BrickMaterial();

protected:
    BaseBrickMaterial *_obj;
    BaseMaterial *_base_obj;

    BrickMaterial(Light const &ambient,
                  Light const &diffuse,
                  Light const &specular,
                  int const &specular_exponent,
                  Light const &transmissive,
                  PREC const &refractive_index,
                  Light const &ambient_edge,
                  Light const &diffuse_edge,
                  Light const &specular_edge,
                  int const &specular_exponent_edge,
                  Light const &transmissive_edge,
                  PREC const &refractive_index_edge,
                  PREC const &scale,
                  PREC const &edge_width,
                  PREC const &edge_height,
                  std::shared_ptr<BumpMapping> const &bump_mapping=nullptr);

    BrickMaterial &operator=(BrickMaterial const &) = delete;
    BrickMaterial &operator=(BrickMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial **_dev_ptr;
    bool _need_upload;
};

#endif // PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP