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
	RecordHit temp_hit;
	bool hits = false;
	
	for (int i = 0; i < m_size; ++i)
	{
		if (m_shapes[i]->hitIntersect(ray, tmin, tmax, temp_hit) && (temp_hit.rayParameter < tmax))
		{
			hits = true;
			tmax = temp_hit.rayParameter;
			hit = temp_hit;
		}
	}
	return hits;
}