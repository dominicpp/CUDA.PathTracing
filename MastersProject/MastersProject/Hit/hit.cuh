#pragma once

#include "../Utils/vec3.cuh"
#include "../Ray/ray.cuh"

class Material;

struct RecordHit final {
	float rayParameter;
	Vec3 positionHit;
	Vec3 normalVector;
	Material* material;
};

class Hit {
public:
	Hit() = default;

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const = 0;
};