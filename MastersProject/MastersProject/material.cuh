#pragma once

#include "hit.cuh"

struct RecordHit;

class Material
{
public:
	Material() = default;
	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& attenuation, Ray& scattered) const = 0;
};

class Lambertian : public Material
{
public:
	Vec3 albedo;
	Lambertian(const Vec3& albedo) : albedo(albedo) {}

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 target = record.positionHit + record.normalVector + random_in_unit_sphere();
		scattered = Ray(record.positionHit, target - record.positionHit);
		attenuation = albedo;
		return true;
	}
};

class Metal : public Material
{
public:
	Vec3 albedo;
	int fuzz;
	Metal(const Vec3& albedo, double f) : albedo(albedo), fuzz(f < 1 ? f : 1) {}

	__host__ __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& record, Vec3& attenuation, Ray& scattered) const
	{
		Vec3 reflected = reflect(unit_vector(ray.getDirection()), record.normalVector);
		scattered = Ray(record.positionHit, reflected + fuzz*random_in_unit_sphere());
		attenuation = albedo;
		return (dot(scattered.getDirection(), record.normalVector) > 0);
	}
};