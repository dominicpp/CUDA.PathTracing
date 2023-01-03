#pragma once

#include "../Utils/vec3.cuh"

class Ray
{
	Vec3 origin, direction;
public:
	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(const Vec3& orig, const Vec3& dir) : origin(orig), direction(dir) {}

	__host__ __device__ Vec3 getOrigin() const { return origin; }
	__host__ __device__ Vec3 getDirection() const { return direction; }

	__host__ __device__ Vec3 pointAt(float t) const { return origin + (t * direction); }
};