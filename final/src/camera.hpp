#ifndef PX_CG_CAMERA_HPP
#define PX_CG_CAMERA_HPP

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#ifdef GLM_FORCE_RADIANS
#define GLM_ANG(ang) ang
#else
#define GLM_ANG(ang) glm::radians(ang)
#endif

namespace px {
class Camera;
}

class px::Camera
{
public:
    static const glm::mat4 IDENTITY_MAT4;

    static const float FOV;
    static const float NEAR_CLIP;
    static const float FAR_CLIP;

    glm::vec3 eye;

    int width;
    int height;
    float near_clip;
    float far_clip;

public:
    Camera();
    void freeze(bool enable);
    void setFov(float fov);
    void setAng(float yaw, float pitch);
    void reset(float x, float y, float z, float yaw, float pitch);

    Camera &zoom(float dfov);
    Camera &pitch(float dpitch);
    Camera &yaw(float dyaw);
    Camera &roll(float droll);

    inline glm::vec3 &camDir() {return strafe;}
    inline glm::vec3 &camRight() {return forward;}

    void updateView();
    void updateProj();

    inline const glm::mat4 &viewMat() const { return view; }
    inline const glm::mat4 &projMat() const { return proj; }

protected:
    glm::mat4 proj, view;

    glm::vec3 strafe;   // first col of view mat
    glm::vec3 forward;  // negative of the 3rd col of view mat

    bool frozen;
private:
    float _pitch, _yaw, _roll;
    float _fov;
};

#endif
