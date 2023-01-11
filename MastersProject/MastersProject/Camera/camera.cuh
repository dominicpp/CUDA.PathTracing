#pragma once

#include "../Ray/ray.cuh"

class Camera
{
    Vec3 m_horizontal, m_vertical, m_origin;

public:
	Camera() = default;
    Camera(double width, double height)
    {
        m_horizontal = Vec3(width, 0.0, 0.0);
        m_vertical = Vec3(0.0, height, 0.0);
        m_origin = Vec3(0.0, 0.0, 0.0);
    }

    Vec3 direction(float width, float height)
    {
        Vec3 cameraPos(0.0, 0.2, 0.8);
        Vec3 corner = m_origin - (m_horizontal / 2) - (m_vertical / 2);
        auto x = width * m_horizontal;
        auto y = height * m_vertical;
        return (corner - cameraPos) + (x + y);
    }

    Ray generateRay(float width, float height) { return Ray(m_origin, direction(width, height)); }
};