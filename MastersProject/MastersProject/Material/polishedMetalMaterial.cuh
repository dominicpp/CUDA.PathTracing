#pragma once

#include "../Material/material.cuh"

// Source 1: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, S. Marschner, [Book] “Fundamentals of Computer Graphics,” 3rd ed., 
// A K Peters / CRC Press, 2009, isbn: 9781568814698.
// Source 3: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

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

    // law of reflection with a scatter factor mulitplied by a random point
    Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
    if (m_scatter_factor != 0.0)
        scattered = Ray(hit.positionHit, reflectionDirection + m_scatter_factor * randomPoints);
    return true;
}