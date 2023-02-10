#pragma once

#include "../Ray/ray.cuh"

class Material;

struct RecordHit final
{
	float rayParameter;
	Vec3 positionHit;
	Vec3 normalVector;
	Material* material;
};

class Shape
{
public:
	Shape() = default;

	virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const = 0;
};