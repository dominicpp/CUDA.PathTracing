#pragma once

#include "../Material/material.cuh"

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612, pp. 174-175

class Mirror : public Material
{
    Vec3 m_albedo;

public:
    __device__ Mirror() = default;
    __device__ Mirror(const Vec3& albedo) : m_albedo(albedo) {}

    __device__ virtual Vec3 albedo() const override;
    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const override;
};

__device__ Vec3 Mirror::albedo() const { return m_albedo; }

__device__ bool Mirror::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const
{
    Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
    scattered = Ray(hit.positionHit, reflectionDirection);
    if (dotProduct(scattered.getDirection(), hit.normalVector) > 0) return true;
    else return false;
}