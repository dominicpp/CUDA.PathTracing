#pragma once

#include "material.cuh"

class Mirror : public Material
{
public:
	Vec3 albedo;
	Mirror() = default;
	Mirror(const Vec3& albedo) : albedo(albedo) {}
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const override;
};

bool Mirror::scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const
{
	Vec3 reflected = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), record.normalVector) * record.normalVector;
	scattered = Ray(record.positionHit, reflected);
	weakening = albedo;
	return (dotProduct(scattered.getDirection(), record.normalVector) > 0);
}