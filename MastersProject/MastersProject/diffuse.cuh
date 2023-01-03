#pragma once

#include "material.cuh"

class Diffuse : public Material
{
public:
	Vec3 albedo;
	Diffuse() = default;
	Diffuse(const Vec3& albedo) : albedo(albedo) {}
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const override;
};

bool Diffuse::scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const
{
	// https://karthikkaranth.me/blog/generating-random-points-in-a-sphere/
	// Monte Carlo Integration!
	double sum, xRnd, yRnd, zRnd;
	do {
		xRnd = random_double() * 2 - 1.0;
		yRnd = random_double() * 2 - 1.0;
		zRnd = random_double() * 2 - 1.0;
		sum = pow(xRnd, 2) + pow(yRnd, 2) + pow(zRnd, 2);
	} while (sum >= 1.0);

	Vec3 target = record.positionHit + normalize(record.normalVector + Vec3(xRnd, yRnd, zRnd));
	scattered = Ray(record.positionHit, target - record.positionHit);
	weakening = albedo;
	return true;
}