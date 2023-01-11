#pragma once

#include "../Hit/hit.cuh"

class Plane : public Hit
{
public:
	Vec3 m_normalVector, m_anchorPoint;

	Plane(Vec3 normalVector, Vec3 anchorPoint) : m_anchorPoint(anchorPoint), m_normalVector(normalVector) {}

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
};

bool Plane::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
{
	float t = (dotProduct(m_anchorPoint, m_normalVector) - dotProduct(m_normalVector, ray.getOrigin()) / dotProduct(ray.getDirection(), m_normalVector));
	if (tmin < t && t < tmax)
	{
		hit.rayParameter = t;
		hit.positionHit = ray.pointAt(hit.rayParameter);
		return true;
	}
	return false;
}