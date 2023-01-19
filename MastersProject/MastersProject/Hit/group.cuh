#pragma once

#include "hit.cuh"

class Group : public Hit
{
	Hit** m_shapes;
	int m_size;

public:
	Group() = default;
	Group(Hit** shapes, int size) : m_shapes(shapes), m_size(size) {};

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
};

bool Group::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
{
    RecordHit tempHit;
    int closestHit = m_size;

    for (int i = 0; i < m_size; ++i)
    {
        if (m_shapes[i]->hitIntersect(ray, tmin, tmax, tempHit) && tempHit.rayParameter < tmax)
        {
            tmax = tempHit.rayParameter;
            closestHit = i;
        }
    }

    if (closestHit != m_size) {
        hit = tempHit;
        return true;
    }
    return false;
}