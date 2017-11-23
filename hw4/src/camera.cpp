#include "camera.hpp"

#include <glm/gtc/matrix_transform.hpp>

using namespace px;

const float Camera::CAM_POS[3] = {0.0f, 0.0f, 3.0f};
const float Camera::CAM_DIR[3] = {0.0f, 0.0f,-1.0f};
const float Camera::CAM_UP[3]  = {0.0f, 1.0f, 0.0f};
const float Camera::CAM_RIGHT[3] = {1.0f, 0.0f, 0.0f};

Camera::Camera()
    : cam_pos(CAM_POS[0], CAM_POS[1], CAM_POS[2]),
      cam_dir(CAM_DIR[0], CAM_DIR[1], CAM_DIR[2]),
      cam_up(CAM_UP[0], CAM_UP[1], CAM_UP[2]),
      cam_right(CAM_RIGHT[0], CAM_RIGHT[1], CAM_RIGHT[2]),
      yaw(-90), pitch(0),
      fov(45.0f), frozen(false)
{}

void Camera::init()
{
    frozen = false;

    updateProjMat();
    updateViewMat();
}

void Camera::reset()
{
    cam_pos.x = CAM_POS[0];
    cam_pos.y = CAM_POS[1];
    cam_pos.z = CAM_POS[2];
    cam_dir.x = CAM_DIR[0];
    cam_dir.y = CAM_DIR[1];
    cam_dir.z = CAM_DIR[2];
    cam_up.x  = CAM_UP[0];
    cam_up.y  = CAM_UP[1];
    cam_up.z  = CAM_UP[2];

    cam_right = glm::cross(cam_up, cam_dir);
    cam_right /= glm::length(cam_right);

    yaw = -90.0f;
    pitch = 0;

    fov = 45.0f;

    init();
}

void Camera::freeze(bool enable)
{
    frozen = enable;
}

void Camera::setFov(float fov_deg)
{
    if (frozen) return;

    fov = fov_deg;
//    if (fov < 1.0f)
//        fov = 1.0f;
//    else if (fov > 45.0f)
//        fov = 45.0f;
    updateProjMat();
}

void Camera::zoom(float d_fov_degree)
{
    if (frozen) return;

    fov += d_fov_degree;
    if (fov < 1.0f)
        fov = 1.0f;
    else if (fov > 45.0f)
        fov = 45.0f;
    updateProjMat();
}

void Camera::updateYaw(float d_yaw)
{
    if (frozen) return;

    yaw += d_yaw;
    if (yaw > 360.0f)
        yaw -= 360.f;
    else if (yaw < -360.f)
        yaw += 360.f;

    auto c = std::cos(glm::radians(pitch));
    cam_dir.x = std::cos(glm::radians(yaw)) * c;
    cam_dir.z = std::sin(glm::radians(yaw)) * c;
    cam_dir /= glm::length(cam_dir);
}

void Camera::_updateCamDir()
{
    auto c = std::cos(glm::radians(pitch));
    cam_dir.x = std::cos(glm::radians(yaw)) * c;
    cam_dir.y = std::sin(glm::radians(pitch));
    cam_dir.z = std::sin(glm::radians(yaw)) * c;
    cam_dir /= glm::length(cam_dir);
}

void Camera::updateAng(float d_yaw, float d_pitch)
{
    if (frozen) return;

    yaw += d_yaw;
    pitch += d_pitch;

    if (yaw > 360.0f)
        yaw -= 360.f;
    else if (yaw < -360.f)
        yaw += 360.f;

    if (pitch > 89.0f)
        pitch = 89.0f;
    else if (pitch < -89.0f)
        pitch = -89.0f;

    _updateCamDir();
}

void Camera::setAng(float y, float p)
{
    if (frozen) return;

    yaw = y;
    pitch = p;

    if (yaw > 360.0f)
        yaw -= 360.f;
    else if (yaw < -360.f)
        yaw += 360.f;

    if (pitch > 89.0f)
        pitch = 89.0f;
    else if (pitch < -89.0f)
        pitch = -89.0f;

    _updateCamDir();
}

void Camera::updateProjMat()
{
#ifdef GLM_FORCE_RADIANS
    proj = glm::perspectiveFov(glm::radians(fov),
                               static_cast<float>(width),
                               static_cast<float>(height),
                               0.1f, 25.0f);
#else
    proj = glm::perspectiveFov(fov,
                               static_cast<float>(width),
                               static_cast<float>(height),
                               0.1f, 150.0f);
#endif
}

void Camera::updateViewMat()
{
    look_at = cam_pos + cam_dir;
    view = glm::lookAt(cam_pos, look_at, cam_up);
    cam_right.x = view[0][0];
    cam_right.y = view[1][0];
    cam_right.z = view[2][0];
}
