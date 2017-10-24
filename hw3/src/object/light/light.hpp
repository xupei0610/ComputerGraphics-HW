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

class px::BaseLight
{
protected:
    BaseLight *dev_ptr;
public:
    enum class Type
    {
        PointLight,
        DirectionalLight,
        AreaLight
    };

    inline BaseLight *devPtr() { return dev_ptr; }
    virtual void up2Gpu() = 0;
    virtual void clearGpuData() = 0;

    PX_CUDA_CALLABLE
    virtual double attenuate(double const &x, double const &y, double const &z) const = 0;
    __device__
    virtual Direction dirFromDevice(double const &x, double const &y, double const &z, double &dist) const = 0;
    virtual Direction dirFromHost(double const &x, double const &y, double const &z, double &dist) const = 0;
    PX_CUDA_CALLABLE
    virtual Type type() const = 0;

    PX_CUDA_CALLABLE
    inline double attenuate(Point const & p) const
    {
        return attenuate(p.x, p.y, p.z);
    }
    __device__
    inline Direction dirFromDevice(Point const &p, double &dist) const
    {
        return dirFromDevice(p.x, p.y, p.z, dist);
    }
    inline Direction dirFromHost(Point const &p, double &dist) const
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
    double attenuate(double const &x, double const &y, double const &z) const override;
    __device__
    Direction dirFromDevice(double const &x, double const &y, double const &z, double &dist) const override;
    Direction dirFromHost(double const &x, double const &y, double const &z, double &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(double const &x, double const &y, double const &z, double &dist) const;
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
    ~DirectionalLight();
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
    double attenuate(double const &x, double const &y, double const &z) const override;
    __device__
    Direction dirFromDevice(double const &x, double const &y, double const &z, double &dist) const override;
    Direction dirFromHost(double const &x, double const &y, double const &z, double &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(double const &x, double const &y, double const &z, double &dist) const;
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
    ~PointLight();
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
                                             double const &half_angle1,
                                             double const &half_angle2,
                                             double const &falloff);
    PX_CUDA_CALLABLE
    double attenuate(double const &x, double const &y, double const &z) const override;
    __device__
    Direction dirFromDevice(double const &x, double const &y, double const &z, double &dist) const override;
    Direction dirFromHost(double const &x, double const &y, double const &z, double &dist) const override;
    PX_CUDA_CALLABLE
    Direction dirFrom(double const &x, double const &y, double const &z, double &dist) const;
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
    void setAngles(double const &half_angle1, double const &half_angle2);
    void setFalloff(double const &falloff);

    inline Point const & position() const noexcept { return _position; }
    inline Direction const & direction() const noexcept { return _direction; }
    inline double const & innerHalfAngle() const noexcept { return _inner_ha; }
    inline double const & outerHalfAngle() const noexcept { return _outer_ha; }
    inline double const & falloff() const noexcept { return _falloff; }

    PX_CUDA_CALLABLE
    ~SpotLight();
    PX_CUDA_CALLABLE
    SpotLight(Light const &light,
              Point const &pos,
              Direction const &direction,
              double const &half_angle1,
              double const &half_angle2,
              double const &falloff);
protected:
    Point _position;
    Direction _direction;
    double _inner_ha;
    double _outer_ha;
    double _falloff;
    double _inner_ha_cosine;
    double _outer_ha_cosine;
    double _multiplier; // 1.0 / (_outer_ha_cosine - inner_ha_cosine)

    bool _need_upload;
};

class px::AreaLight : public BaseLight
{
public:
    Type const TYPE;

    static std::shared_ptr<BaseLight> create(Light const &light,
                                             Point const &center,
                                             double const &radius);

    PX_CUDA_CALLABLE
    double attenuate(double const &x, double const &y, double const &z) const override;
    __device__
    virtual Direction dirFromDevice(double const &x, double const &y, double const &z, double &dist) const;
    virtual Direction dirFromHost(double const &x, double const &y, double const &z, double &dist) const;
    PX_CUDA_CALLABLE
    Type type() const override
    {
        return TYPE;
    }

    void up2Gpu() override;
    void clearGpuData() override;

    inline Point const & center() const noexcept {return _center;}
    inline double const & radius() const noexcept { return _radius; }

    void setCenter(Point const &center);
    void setRadius(double const &radius);

    PX_CUDA_CALLABLE
    ~AreaLight();
    PX_CUDA_CALLABLE
    AreaLight(Light const &light,
              Point const &center,
              double const &radius);
protected:
    Point _center;
    double _radius;

    bool _need_upload;
};


#endif // PX_CG_OBJECT_LIGHT_LIGHT_HPP