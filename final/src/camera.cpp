#include "camera.hpp"

#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <iostream>

using namespace px;

const glm::mat4 Camera::IDENTITY_MAT4(1.f);
const float Camera::FOV = 45.f;
const float Camera::NEAR_CLIP = 0.01f;
const float Camera::FAR_CLIP = 150.f;

Camera::Camera()
        : eye(0, 0, -1), near_clip(NEAR_CLIP), far_clip(FAR_CLIP), frozen(false),
          _pitch(0), _yaw(-90.f), _roll(0), _fov(FOV)
{
    updateView();
}


void Camera::freeze(bool enable)
{
    frozen = enable;
}


void Camera::setFov(float fov)
{
    if (frozen) return;
    _fov = fov;
}

void Camera::setAng(float yaw, float pitch)
{
    if (frozen) return;
    _yaw = yaw;
    _pitch = pitch;
}

void Camera::reset(float x, float y, float z, float yaw, float pitch)
{
    if (frozen) return;
    eye.x = x;
    eye.y = y;
    eye.z = z;
    _yaw = yaw;
    _pitch = pitch;
}

Camera &Camera::zoom(float dfov)
{
    if (!frozen)
    {
        _fov += dfov;
        if (_fov > 89.f) _fov = 89.f;
        else if (_fov < 30.f) _fov = 30.f;
    }

    return *this;
}

Camera &Camera::pitch(float dpitch)
{
    if (!frozen)
    {
        _pitch += dpitch;
        if (_pitch > 89.0f)
            _pitch = 89.0f;
        else if (_pitch < -89.0f)
            _pitch = -89.0f;
    }
    return *this;
}

Camera &Camera::yaw(float dyaw)
{
    if (!frozen)
    {
        _yaw += dyaw;
        if (_yaw > 360.0f)
            _yaw -= 360.f;
        else if (_yaw < -360.f)
            _yaw += 360.f;
    }
    return *this;
}

Camera &Camera::roll(float droll)
{
    if (!frozen)
    {
        _roll += droll;
    }
    return *this;
}

void Camera::updateView()
{
//    glm::quat orient = glm::quat(glm::vec3(GLM_ANG(_pitch), GLM_ANG(_yaw), 0));

    glm::quat pitch = glm::angleAxis(GLM_ANG(_pitch), glm::vec3(1, 0, 0));
    glm::quat yaw = glm::angleAxis(GLM_ANG(_yaw), glm::vec3(0, 1, 0));
    glm::quat orient = pitch * yaw;

    glm::mat4 rot = glm::mat4_cast(glm::normalize(orient));
    glm::mat4 trans = glm::translate(IDENTITY_MAT4, -eye);

    view = rot * trans;

    strafe.x = -view[0][2];
    strafe.y = -view[1][2];
    strafe.z = -view[2][2];
    forward.x = view[0][0];
    forward.y = view[1][0];
    forward.z = view[2][0];
}

void Camera::updateProj()
{
    proj = glm::perspectiveFov(GLM_ANG(_fov),
                               static_cast<float>(width),
                               static_cast<float>(height),
                               near_clip, far_clip);
}
