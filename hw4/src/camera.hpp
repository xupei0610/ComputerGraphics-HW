#ifndef PX_CG_CAMERA_HPP
#define PX_CG_CAMERA_HPP

#include <glm/glm.hpp>

#include "option.hpp"

namespace px {

class Camera;

}

class px::Camera
{
public:
    static const float CAM_POS[3];
    static const float CAM_DIR[3];
    static const float CAM_UP[3];
    static const float CAM_RIGHT[3];

    glm::vec3 cam_pos;
    glm::vec3 cam_dir;
    int width;
    int height;

protected:
    glm::mat4 view;
    glm::mat4 proj;

    glm::vec3 cam_up;
    glm::vec3 cam_right;
    glm::vec3 look_at;

    float yaw;
    float pitch;

    float fov;

    bool frozen;
public:
    Camera();

    inline glm::mat4 const &viewMat() const noexcept {return view;}
    inline glm::mat4 const &projMat() const noexcept {return proj;}
    inline glm::vec3 const &camRight() const noexcept {return cam_right;}
    inline glm::vec3 const &lookAt() const noexcept {return look_at;}

    void init();
    void reset();

    void freeze(bool enable);

    void updateProjMat();
    void updateViewMat();

    void setFov(float fov_deg);
    void zoom(float d_fov_degree);
    void updateYaw(float d_yaw);
    void updateAng(float d_yaw, float d_pitch);
    void setAng(float yaw, float pitch);

    ~Camera() = default;
    Camera &operator=(Camera const &) = default;
    Camera &operator=(Camera &&) = default;

private:
    void _updateCamDir();
};

#endif
