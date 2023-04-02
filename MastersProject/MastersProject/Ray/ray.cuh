#pragma once

#include "../Utils/vec3.cuh"

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.

class Ray
{
	Vec3 m_origin, m_direction;

public:
	__device__ Ray() = default;
	__device__ Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_direction(direction) {}

	__device__ Vec3 getOrigin() const { return m_origin; }
	__device__ Vec3 getDirection() const { return m_direction; }

	__device__ Vec3 pointAt(float t) const { return m_origin + (t * m_direction); }
};