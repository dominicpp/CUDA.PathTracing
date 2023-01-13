#pragma once

#include "material.cuh"

class PolishedMetal : public Material
{
	Vec3 m_albedo;
	int m_scatter_factor;

public:
	PolishedMetal() = default;
	PolishedMetal(const Vec3& albedo, double scatter_factor) : m_albedo(albedo), m_scatter_factor(scatter_factor) {}

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const override;
};

bool PolishedMetal::scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const
{
	double xRnd, yRnd, zRnd;
	xRnd = random_double() * 2 - 1.0;
	yRnd = random_double() * 2 - 1.0;
	zRnd = random_double() * 2 - 1.0;

	Vec3 randomPoints(xRnd, yRnd, zRnd);
	Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
	scattered = Ray(hit.positionHit, reflectionDirection + m_scatter_factor * randomPoints);
	weakening = m_albedo;
	if (m_scatter_factor != 0) return (dotProduct(scattered.getDirection(), hit.normalVector) > 0);
	else return (dotProduct(scattered.getDirection(), hit.normalVector) < 1);
}