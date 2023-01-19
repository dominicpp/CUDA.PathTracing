#pragma once

#include "material.cuh"

class Mirror : public Material
{
	Vec3 m_albedo;

public:
	Mirror() = default;
	Mirror(const Vec3& albedo) : m_albedo(albedo) {}

	__host__ __device__ virtual Vec3 albedo() const override;
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const override;
};

Vec3 Mirror::albedo() const { return m_albedo; }

bool Mirror::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const
{
	Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
	scattered = Ray(hit.positionHit, reflectionDirection);
	if (dotProduct(scattered.getDirection(), hit.normalVector) > 0) return true;
	else return false;
}