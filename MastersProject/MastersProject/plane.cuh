#pragma once

#include "hit.cuh"

class Plane : public Hit
{
public:
	Vec3 normalVector, anchorPoint;

	Plane(Vec3 normVec, Vec3 anchorP) : anchorPoint(anchorP), normalVector(normVec) {}

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const override;
};

bool Plane::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const
{
	float t = (dotProduct(anchorPoint, normalVector) - dotProduct(normalVector, ray.getOrigin()) / dotProduct(ray.getDirection(), normalVector));
	if (tmin < t && t < tmax)
	{
		record.rayParameter = t;
		record.positionHit = ray.pointAt(record.rayParameter);
		return true;
	}
	return false;
}