#pragma once

#include "../Hit/hit.cuh"

struct RecordHit;

class Material
{
public:
	Material() = default;

	__host__ __device__ virtual Vec3 albedo() const = 0;

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const = 0;
};