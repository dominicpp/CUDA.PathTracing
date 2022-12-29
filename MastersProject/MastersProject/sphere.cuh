#pragma once

#include "hit.cuh"
#include "ray.cuh"

class Sphere : public Hit {
	Vec3 position;
	float radius;
public:
	Sphere() = default;
	Sphere(Vec3 p, float r) : position(p), radius(r) {};
	__host__ __device__ virtual bool hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const override;
};

bool Sphere::hitIntersect(const Ray& ray, float tmin, float tmax, RecordHit& record) const
{
	Vec3 newPosition = ray.getOrigin() - position;
	float a = dot(ray.getDirection(), ray.getDirection());
	float b = dot(newPosition, ray.getDirection());
	float c = dot(newPosition, newPosition) - pow(radius, 2);
	float discriminant = pow(b, 2) - a * c;
	float t;

	if (discriminant < 0) return false;
	if (discriminant >= 0)
	{
		float t1 = (-(b + sqrt(discriminant))) / a;
		float t2 = (-(b - sqrt(discriminant))) / a;

		if (t1 < t2) t = t1;
		else t = t2; 

		if (t < tmax && t > tmin)
		{
			record.rayParameter = t;
			record.positionHit = ray.pointAt(record.rayParameter);
			record.normalVector = (record.positionHit - position) / radius;
			return true;
		}
	}
}