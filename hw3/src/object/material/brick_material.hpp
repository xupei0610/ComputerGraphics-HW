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

    ~BaseBrickMaterial() = default;

    PX_CUDA_CALLABLE
    int specularExp(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    double refractiveIndex(double const &u, double const &v, double const &w) const override;

    bool onEdge(double const &u, double const &v, double const &w) const noexcept;

protected:
    PX_CUDA_CALLABLE
    Light getAmbient(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getDiffuse(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getSpecular(double const &u, double const &v, double const &w) const override;
    PX_CUDA_CALLABLE
    Light getTransmissive(double const &u, double const &v, double const &w) const override;

    BaseBrickMaterial(Light const &ambient,
                      Light const &diffuse,
                      Light const &specular,
                      int const &specular_exponent,
                      Light const &transmissive,
                      double const &refractive_index,
                      Light const &ambient_edge,
                      Light const &diffuse_edge,
                      Light const &specular_edge,
                      int const &specular_exponent_edge,
                      Light const &transmissive_edge,
                      double const &refractive_index_edge,
                      double const &scale,
                      double const &edge_width,
                      double const &edge_height,
                      const BumpMapping *const &bump_mapping);
    Light _ambient;
    Light _diffuse;
    Light _specular;
    int _specular_exponent;
    Light _transmissive;
    double _refractive_index;

    Light _ambient_edge;
    Light _diffuse_edge;
    Light _specular_edge;
    int _specular_exponent_edge;
    Light _transmissive_edge;
    double _refractive_index_edge;

    double _scale;
    double _edge_width;
    double _edge_height;
};


class px::BrickMaterial : public BaseBrickMaterial
{
public:

    static std::shared_ptr<BaseMaterial> create(Light const &ambient,
                                                Light const &diffuse,
                                                Light const &specular,
                                                int const &specular_exponent,
                                                Light const &transmissive,
                                                double const &refractive_index,
                                                Light const &ambient_edge,
                                                Light const &diffuse_edge,
                                                Light const &specular_edge,
                                                int const &specular_exponent_edge,
                                                Light const &transmissive_edge,
                                                double const &refractive_index_edge,
                                                double const &scale,
                                                double const &edge_width,
                                                double const &edge_height,
                                                std::shared_ptr<BumpMapping> const &bump_mapping=nullptr);
    ~BrickMaterial() = default;

    BaseMaterial *up2Gpu() override;
    void clearGpuData() override ;

    void setAmbient(Light const &ambient);
    void setDiffuse(Light const &diffuse);
    void setSpecular(Light const &specular);
    void setSpecularExp(int const &specular_exp);
    void setTransmissive(Light const &transmissive);
    void setRefractiveIndex(double const &ior);
    void setAmbientEdge(Light const &ambient);
    void setDiffuseEdge(Light const &diffuse);
    void setSpecularEdge(Light const &specular);
    void setSpecularExpEdge(int const &specular_exp);
    void setTransmissiveEdge(Light const &transmissive);
    void setRefractiveIndexEdge(double const &ior);
    void setBumpMapping(std::shared_ptr<BumpMapping> const &bump_mapping);
    void setScale(double const &scale);
    void setEdgeWidth(double const &width);
    void setEdgeHeight(double const &height);

protected:
    BrickMaterial(Light const &ambient,
                  Light const &diffuse,
                  Light const &specular,
                  int const &specular_exponent,
                  Light const &transmissive,
                  double const &refractive_index,
                  Light const &ambient_edge,
                  Light const &diffuse_edge,
                  Light const &specular_edge,
                  int const &specular_exponent_edge,
                  Light const &transmissive_edge,
                  double const &refractive_index_edge,
                  double const &scale,
                  double const &edge_width,
                  double const &edge_height,
                  std::shared_ptr<BumpMapping> const &bump_mapping=nullptr);

    BrickMaterial &operator=(BrickMaterial const &) = delete;
    BrickMaterial &operator=(BrickMaterial &&) = delete;

    std::shared_ptr<BumpMapping> _bump_mapping_ptr;
    BaseMaterial *_dev_ptr;
    bool _need_upload;
};

#endif // PX_CG_OBJECT_MATERIAL_BRICK_MATERIAL_HPP