#pragma once

#include "../Ray/ray.cuh"

class Camera
{
    Vec3 m_horizontal, m_vertical, m_origin, m_cameraPos;

public:
    Camera() = default;
    Camera(float positionX, float positionY)
    {
        m_horizontal = Vec3(positionX, 0.0, 0.0);
        m_vertical = Vec3(0.0, positionY, 0.0);
        m_origin = Vec3(0.0, 0.0, 0.0);
        m_cameraPos = Vec3(0.0, 0.2, 0.8);
    }

    Vec3 direction(float width, float height) const
    {
        Vec3 corner = m_origin - (m_horizontal / 2) - (m_vertical / 2);
        auto x = width * m_horizontal;
        auto y = height * m_vertical;
        return (corner - m_cameraPos) + (x + y);
    }

    Ray generateRay(float width, float height) const { return Ray(m_origin, direction(width, height)); }
};