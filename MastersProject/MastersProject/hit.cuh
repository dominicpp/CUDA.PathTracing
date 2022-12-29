#pragma once

#include "vec3.cuh"
#include "ray.cuh"

struct RecordHit final {
	float rayParameter;
	Vec3 positionHit;
	Vec3 normalVector;
};

class Hit {
public:
	Hit() = default;
	__host__ __device__ virtual bool hitIntersect(const Ray& r, float tmin, float tmax, RecordHit& record) const = 0;
};