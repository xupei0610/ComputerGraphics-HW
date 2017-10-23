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
public:
    enum class Type
    {
        PointLight,
        DirectionalLight,
        AreaLight
    };

    virtual BaseLight *up2Gpu() = 0;
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

    inline Light const &light() const noexcept { return _light; }
    void setLight(Light const &light);

    virtual ~BaseLight() = default;
protected:
    Light _light;

    bool need_upload;

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

    BaseLight *up2Gpu() override;
    void clearGpuData() override;

    inline Direction const & direction() const noexcept { return _dir; }
    void setDirection(Direction const &dir);

    ~DirectionalLight();
protected:
    Direction _dir;
    Direction _neg_dir;

    BaseLight *_dev_ptr;
    bool _need_upload;

    DirectionalLight(Light const &light, Direction const &dir);
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

    BaseLight *up2Gpu() override;
    void clearGpuData() override;

    inline Point const & position() const noexcept { return _position;}
    void setPosition(Point const &p);

    ~PointLight();
protected:
    Point _position;

    BaseLight *_dev_ptr;
    bool _need_upload;

    PointLight(Light const &light, Point const &pos);
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

    BaseLight *up2Gpu() override;
    void clearGpuData() override;

    void setPosition(Point const &pos);
    void setDirection(Direction const &direction);
    void setAngles(double const &half_angle1, double const &half_angle2);
    void setFalloff(double const &falloff);

    inline Point const & position() const noexcept { return _position; }
    inline Direction const & direction() const noexcept { return _direction; }
    inline double const & innerHalfAngle() const noexcept { return _inner_ha; }
    inline double const & outerHalfAngle() const noexcept { return _outer_ha; }
    inline double const & falloff() const noexcept { return _falloff; }

    ~SpotLight();
protected:
    Point _position;
    Direction _direction;
    double _inner_ha;
    double _outer_ha;
    double _falloff;
    double _inner_ha_cosine;
    double _outer_ha_cosine;
    double _multiplier; // 1.0 / (_outer_ha_cosine - inner_ha_cosine)

    BaseLight *_dev_ptr;
    bool _need_upload;

    SpotLight(Light const &light,
              Point const &pos,
              Direction const &direction,
              double const &half_angle1,
              double const &half_angle2,
              double const &falloff);
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

    BaseLight *up2Gpu() override;
    void clearGpuData() override;

    inline Point const & center() const noexcept {return _center;}
    inline double const & radius() const noexcept { return _radius; }

    void setCenter(Point const &center);
    void setRadius(double const &radius);

    ~AreaLight();
protected:
    Point _center;
    double _radius;

    BaseLight *_dev_ptr;
    bool _need_upload;

    AreaLight(Light const &light,
              Point const &center,
              double const &radius);
};

#endif // PX_CG_OBJECT_LIGHT_LIGHT_HPP