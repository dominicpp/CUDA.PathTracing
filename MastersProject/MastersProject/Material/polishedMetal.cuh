#pragma once

#include "material.cuh"

class PolishedMetal : public Material
{
	Vec3 m_albedo;
	float m_scatter_factor;
	double m_pi = 3.14159265358979323846;

public:
	PolishedMetal() = default;
	PolishedMetal(const Vec3& albedo, float scatter_factor) : m_albedo(albedo), m_scatter_factor(scatter_factor) {}

	__host__ __device__ virtual Vec3 albedo() const override;
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const override;
};

Vec3 PolishedMetal::albedo() const { return m_albedo; }

bool PolishedMetal::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const
{
	double xRnd, yRnd, zRnd;
	xRnd = random_double() * 2 - 1.0;
	yRnd = random_double() * 2 - 1.0;
	zRnd = random_double() * 2 - 1.0;

	Vec3 randomPoints(xRnd, yRnd, zRnd);
	Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
	if (m_scatter_factor != 0.0)
		scattered = Ray(hit.positionHit, reflectionDirection + m_scatter_factor * randomPoints);
	if (dotProduct(scattered.getDirection(), hit.normalVector) > 0 && dotProduct(scattered.getDirection(), hit.normalVector) < m_pi 
		|| dotProduct(scattered.getDirection(), hit.normalVector) < 1) return true;
	return false;
}