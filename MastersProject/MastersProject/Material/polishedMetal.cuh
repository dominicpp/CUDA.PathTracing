#pragma once

#include "material.cuh"

class PolishedMetal : public Material
{
	Vec3 albedo;
	int scatter_factor;

public:
	PolishedMetal() = default;
	PolishedMetal(const Vec3& albedo, double sf) : albedo(albedo), scatter_factor(sf) {}
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const override;
};

bool PolishedMetal::scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const
{
	double xRnd, yRnd, zRnd;
	xRnd = random_double() * 2 - 1.0;
	yRnd = random_double() * 2 - 1.0;
	zRnd = random_double() * 2 - 1.0;

	Vec3 random(xRnd, yRnd, zRnd);
	Vec3 reflected = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), record.normalVector) * record.normalVector;
	scattered = Ray(record.positionHit, reflected + scatter_factor * random);
	weakening = albedo;
	if (scatter_factor != 0) return (dotProduct(scattered.getDirection(), record.normalVector) > 0);
	else return (dotProduct(scattered.getDirection(), record.normalVector) < 1);
}