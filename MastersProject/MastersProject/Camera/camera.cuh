#pragma once

#include "../Ray/ray.cuh"

// Source 1: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

class Camera
{
    Vec3 m_horizontal, m_vertical, m_origin, m_cameraPos;

public:
    __device__ Camera() = default;
    __device__ Camera(float positionX, float positionY)
        : m_horizontal(positionX, 0.0f, 0.0f)
        , m_vertical(0.0f, positionY, 0.0f)
        , m_origin(0.0f, 0.0f, 0.0f)
        , m_cameraPos(0.0f, 0.2f, 0.8f)
        {}

    __device__ Vec3 direction(float width, float height) const
    {
        Vec3 corner = m_origin - (m_horizontal / 2) - (m_vertical / 2);
        auto x = width * m_horizontal;
        auto y = height * m_vertical;
        return (corner - m_cameraPos) + (x + y);
    }

    __device__ Ray generateRay(float width, float height) const { return Ray(m_origin, direction(width, height)); }
};