#pragma once

#include "material.cuh"

class Diffuse : public Material
{
	Vec3 m_albedo;

public:
	Diffuse() = default;
	Diffuse(const Vec3& albedo) : m_albedo(albedo) {}

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const override;
};

bool Diffuse::scatteredRay(const Ray& ray, const RecordHit& hit, Vec3& weakening, Ray& scattered) const
{
	double sum, xRnd, yRnd, zRnd;
	do {
		xRnd = random_double() * 2 - 1.0;
		yRnd = random_double() * 2 - 1.0;
		zRnd = random_double() * 2 - 1.0;
		sum = pow(xRnd, 2) + pow(yRnd, 2) + pow(zRnd, 2);
	} while (sum >= 1.0);

	Vec3 target = hit.positionHit + normalize(hit.normalVector + Vec3(xRnd, yRnd, zRnd));
	scattered = Ray(hit.positionHit, target - hit.positionHit);
	weakening = m_albedo;
	return true;
}