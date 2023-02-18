#pragma once

#include "../Material/material.cuh"

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612, p. 175

#define PI 3.14159265358979323846

class PolishedMetal : public Material
{
    Vec3 m_albedo;
    float m_scatter_factor;

public:
    __device__ PolishedMetal() = default;
    __device__ PolishedMetal(const Vec3& albedo, float scatter_factor) : m_albedo(albedo), m_scatter_factor(scatter_factor) {}

    __device__ virtual Vec3 albedo() const override;
    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const override;
};

__device__ Vec3 PolishedMetal::albedo() const { return m_albedo; }

__device__ bool PolishedMetal::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const
{
    double xRnd, yRnd, zRnd;
    xRnd = curand_uniform(state) * 2 - 1.0;
    yRnd = curand_uniform(state) * 2 - 1.0;
    zRnd = curand_uniform(state) * 2 - 1.0;
    Vec3 randomPoints(xRnd, yRnd, zRnd);

    Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
    if (m_scatter_factor != 0.0)
        scattered = Ray(hit.positionHit, reflectionDirection + m_scatter_factor * randomPoints);
    if (dotProduct(scattered.getDirection(), hit.normalVector) > 0 && dotProduct(scattered.getDirection(), hit.normalVector) < PI
        || dotProduct(scattered.getDirection(), hit.normalVector) < 1) return true;
    return false;
}