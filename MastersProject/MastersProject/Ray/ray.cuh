#pragma once

#include "../Utils/vec3.cuh"

class Ray
{
	Vec3 m_origin, m_direction;

public:
	__host__ __device__ Ray() = default;
	__host__ __device__ Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_direction(direction) {}

	__host__ __device__ Vec3 getOrigin() const { return m_origin; }
	__host__ __device__ Vec3 getDirection() const { return m_direction; }

	__host__ __device__ Vec3 pointAt(float t) const { return m_origin + (t * m_direction); }
};