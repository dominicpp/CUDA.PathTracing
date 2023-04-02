#pragma once

#include "../Hit/shape.cuh"

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.

class Group : public Shape
{
    Shape** m_shapes;
    int m_size;

public:
    __device__ Group() = default;
    __device__ Group(Shape** shapes, int size) : m_shapes(shapes), m_size(size) {};

    __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
};

// determine whether a given ray intersects with any shapes in a group
__device__ bool Group::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
{
    RecordHit closestHit;
    int closestShapeIndex = m_size;

    for (int i = 0; i < m_size; ++i)
    {
        RecordHit tempHit;
        if (m_shapes[i]->hitIntersect(ray, tmin, tmax, tempHit) && tempHit.rayParameter < tmax)
        {
            tmax = tempHit.rayParameter;
            closestHit = tempHit;
            closestShapeIndex = i;
        }
    }

    if (closestShapeIndex != m_size) {
        hit = closestHit;
        return true;
    }
    return false;
}