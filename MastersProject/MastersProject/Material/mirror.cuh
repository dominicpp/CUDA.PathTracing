#pragma once

#include "material.cuh"

class Mirror : public Material
{
	Vec3 m_albedo;

public:
	Mirror() = default;
	Mirror(const Vec3& albedo) : m_albedo(albedo) {}

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const override;
};

bool Mirror::scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const
{
	Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
	scattered = Ray(hit.positionHit, reflectionDirection);
	weakening = m_albedo;
	if (dotProduct(scattered.getDirection(), hit.normalVector) > 0) return true;
	else return false;
}