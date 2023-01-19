#pragma once

#include "../Hit/hit.cuh"
#include "../Ray/ray.cuh"

class Material;

class Sphere : public Hit {
	Vec3 m_position;
	float m_radius;
	Material* m_material;

public:
	Sphere() = default;
	Sphere(Vec3 position, float radius, Material* material) : m_position(position), m_radius(radius), m_material(material) {};

	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const override;
};

bool Sphere::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& hit) const
{
	Vec3 newPosition = ray.getOrigin() - m_position;
	float a = dotProduct(ray.getDirection(), ray.getDirection());
	float b = 2 * dotProduct(newPosition, ray.getDirection());
	float c = dotProduct(newPosition, newPosition) - pow(m_radius, 2);
	float discriminant = pow(b, 2) - (4 * a * c);
	float t;

	if (discriminant < 0) return false;
	if (discriminant >= 0)
	{
		float t1 = (-(b + sqrt(discriminant))) / (2 * a);
		float t2 = (-(b - sqrt(discriminant))) / (2 * a);

		t1 < t2 ? t = t1 : t = t2;

		if (tmin < t && t < tmax)
		{
			hit.rayParameter = t;
			hit.positionHit = ray.pointAt(hit.rayParameter);
			hit.normalVector = (hit.positionHit - m_position) / m_radius;
			hit.material = m_material;
			return true;
		}
	}
}