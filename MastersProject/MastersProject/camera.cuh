#pragma once

#include "ray.cuh"

class Camera
{
    Vec3 horizontal, vertical, origin;
public:
	Camera() = default;
    Camera(double x, double y)
    {
        horizontal = Vec3(x, 0.0, 0.0);
        vertical = Vec3(0.0, y, 0.0);
        origin = Vec3(0.0, 0.0, 0.0);
    }

    Vec3 direction(float x, float y)
    {
        return (origin - (horizontal / 2) - (vertical / 2) - Vec3(0, 0.2, 0.8)) + (x * horizontal) + (y * vertical);
    }

    Ray generateRay(float x, float y) 
    { 
        return Ray(origin, direction(x, y));
    }
};