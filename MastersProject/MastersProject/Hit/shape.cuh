#pragma once

#include "../Ray/ray.cuh"

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.

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