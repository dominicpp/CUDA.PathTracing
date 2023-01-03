#pragma once

#include "hit.cuh"

class Hitlist : public Hit
{
public:
	Hit** list;
	int list_size;

	Hitlist() = default;
	Hitlist(Hit** l, int n) : list(l), list_size(n) {};
	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const override;
};

bool Hitlist::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const
{
	RecordHit t;
	bool hit = false;
	float m = tmax;
	for (int i = 0; i < list_size; i++)
	{
		if (list[i]->hitIntersect(ray, tmin, m, t))
		{
			hit = true;
			m = t.rayParameter;
			record = t;
		}
	}
	return hit;
}