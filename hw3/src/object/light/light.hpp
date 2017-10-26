#ifndef PX_CG_OBJECT_LIGHT_LIGHT_HPP
#define PX_CG_OBJECT_LIGHT_LIGHT_HPP

#include "object/base_object.hpp"

namespace px
{
class BaseLight;
class DirectionalLight;
class PointLight;
class SpotLight;
class AreaLight;

// TODO Image-based Lighting (environment maps)
}

#include <iostream>
class px::BaseLight
{
protected:
    BaseLight **dev_ptr;
public:
    enum class Type
    {
        PointLight,
        DirectionalLight,
        AreaLight
    };

    inline BaseLight **devPtr() { return dev_ptr; }
    virtual void up2Gpu() = 0;
    virtual void clearGpuData() = 0;

    PX_CUDA_CALLABLE
    virtual PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const = 0;
    __device__
    virtual Direction dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const = 0;
    virtual Direction dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const = 0;
    PX_CUDA_CALLABLE
    virtual Type type() const = 0;

    PX_CUDA_CALLABLE
    inline PREC attenuate(Point const & p) const
    {
        return attenuate(p.x, p.y, p.z);
    }
    __device__
    inline Direction dirFromDevice(Point const &p, PREC &dist) const
    {
        return dirFromDevice(p.x, p.y, p.z, dist);
    }
    inline Direction dirFromHost(Point const &p, PREC &dist) const
    {
        return dirFromHost(p.x, p.y, p.z, dist);
    }
    PX_CUDA_CALLABLE
    inline Light const &light() const noexcept { return _light; }

    void setLight(Light const &light);

    PX_CUDA_CALLABLE
    virtual ~BaseLight() = default;
protected:
    Light _light;

    bool need_upload;

    PX_CUDA_CALLABLE
    BaseLight(Light const &light);
    BaseLight &operator=(BaseLight const &) = delete;
    BaseLight &operator=(BaseLight &&) = delete;
};

class px::DirectionalLight : public BaseLight
{
public:
    Type const TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light, Direction const &dir);

    PX_CUDA_CALLABLE
    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    __device__
    Direction dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    Direction dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const;
    PX_CUDA_CALLABLE
    Type type() const override
    {
        return TYPE;
    }

    void up2Gpu() override;
    void clearGpuData() override;

    inline Direction const & direction() const noexcept { return _dir; }
    void setDirection(Direction const &dir);

    PX_CUDA_CALLABLE
    ~DirectionalLight() = default;
    PX_CUDA_CALLABLE
    DirectionalLight(Light const &light, Direction const &dir);
protected:
    Direction _dir;
    Direction _neg_dir;

    bool _need_upload;

};

class px::PointLight : public BaseLight
{
public:
    Type const TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light, Point const &pos);

    PX_CUDA_CALLABLE
    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    __device__
    Direction dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    Direction dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const;
    PX_CUDA_CALLABLE
    Type type() const override
    {
        return TYPE;
    }

    void up2Gpu() override;
    void clearGpuData() override;

    inline Point const & position() const noexcept { return _position;}
    void setPosition(Point const &p);

    PX_CUDA_CALLABLE
    ~PointLight() = default;
    PX_CUDA_CALLABLE
    PointLight(Light const &light, Point const &pos);
protected:
    Point _position;

    bool _need_upload;
};

class px::SpotLight : public BaseLight
{
public:
    Type const TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &pos,
                                             Direction const &direction,
                                             PREC const &half_angle1,
                                             PREC const &half_angle2,
                                             PREC const &falloff);
    PX_CUDA_CALLABLE
    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    __device__
    Direction dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    Direction dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const;
    PX_CUDA_CALLABLE
    Type type() const override
    {
        return TYPE;
    }

    void up2Gpu() override;
    void clearGpuData() override;

    void setPosition(Point const &pos);
    void setDirection(Direction const &direction);
    PX_CUDA_CALLABLE
    void setAngles(PREC const &half_angle1, PREC const &half_angle2);
    void setFalloff(PREC const &falloff);

    inline Point const & position() const noexcept { return _position; }
    inline Direction const & direction() const noexcept { return _direction; }
    inline PREC const & innerHalfAngle() const noexcept { return _inner_ha; }
    inline PREC const & outerHalfAngle() const noexcept { return _outer_ha; }
    inline PREC const & falloff() const noexcept { return _falloff; }

    PX_CUDA_CALLABLE
    ~SpotLight() = default;
    PX_CUDA_CALLABLE
    SpotLight(Light const &light,
              Point const &pos,
              Direction const &direction,
              PREC const &half_angle1,
              PREC const &half_angle2,
              PREC const &falloff);
protected:
    Point _position;
    Direction _direction;
    PREC _inner_ha;
    PREC _outer_ha;
    PREC _falloff;
    PREC _inner_ha_cosine;
    PREC _outer_ha_cosine;
    PREC _multiplier; // 1.0 / (_outer_ha_cosine - inner_ha_cosine)

    bool _need_upload;
};

class px::AreaLight : public BaseLight
{
public:
    Type const TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &center,
                                             PREC const &radius);

    PX_CUDA_CALLABLE
    PREC attenuate(PREC const &x, PREC const &y, PREC const &z) const override;
    __device__
    virtual Direction dirFromDevice(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const;
    virtual Direction dirFromHost(PREC const &x, PREC const &y, PREC const &z, PREC &dist) const;
    PX_CUDA_CALLABLE
    Type type() const override
    {
        return TYPE;
    }

    void up2Gpu() override;
    void clearGpuData() override;

    inline Point const & center() const noexcept {return _center;}
    inline PREC const & radius() const noexcept { return _radius; }

    void setCenter(Point const &center);
    void setRadius(PREC const &radius);

    PX_CUDA_CALLABLE
    ~AreaLight() = default;
    PX_CUDA_CALLABLE
    AreaLight(Light const &light,
              Point const &center,
              PREC const &r);
protected:
    Point _center;
    PREC _radius;

    bool _need_upload;
};


#endif // PX_CG_OBJECT_LIGHT_LIGHT_HPP