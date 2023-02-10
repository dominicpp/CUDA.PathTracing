#pragma once

#include "../Utils/vec3.cuh"

class Ray
{
	Vec3 m_origin, m_direction;

public:
	Ray() = default;
	Ray(const Vec3& origin, const Vec3& direction) : m_origin(origin), m_direction(direction) {}

	Vec3 getOrigin() const { return m_origin; }
	Vec3 getDirection() const { return m_direction; }

	Vec3 pointAt(float t) const { return m_origin + (t * m_direction); }
};