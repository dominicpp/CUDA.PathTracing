#pragma once

#include "hit.cuh"

struct RecordHit;

class Material
{
public:
	Material() = default;
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& weakening, Ray& scattered) const = 0;
};