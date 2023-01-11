#pragma once

#include "hit.cuh"
#include <iostream>
#include <vector>

class Group : public Hit
{
	std::vector<Hit*> shapes;

public:
	Group() = default;
	Group(std::vector<Hit*> s) : shapes(s) {};

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const override;
};

bool Group::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const
{
	RecordHit h;
	bool hit = false;

	for (const auto& shape : shapes)
	{
		if (shape->hitIntersect(ray, tmin, tmax, h))
		{
			hit = true;
			tmax = h.rayParameter;
			record = h;
		}
	}
	return hit;
}