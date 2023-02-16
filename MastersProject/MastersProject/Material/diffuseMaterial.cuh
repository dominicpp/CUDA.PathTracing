#pragma once

#include "../Material/material.cuh"

class Diffuse : public Material
{
    Vec3 m_albedo;

public:
    __device__ Diffuse() = default;
    __device__ Diffuse(const Vec3& albedo) : m_albedo(albedo) {}

    __device__ virtual Vec3 albedo() const override;
    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const override;
};

__device__ Vec3 Diffuse::albedo() const { return m_albedo; }

__device__ bool Diffuse::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const
{
    // Monte Carlo
    double sum, xRnd, yRnd, zRnd;
    sum = xRnd = yRnd = zRnd = 1.0;
    while (sum >= 1.0) 
    {
        xRnd = curand_uniform(state) * 2 - 1.0;
        yRnd = curand_uniform(state) * 2 - 1.0;
        zRnd = curand_uniform(state) * 2 - 1.0;
        sum = pow(xRnd, 2) + pow(yRnd, 2) + pow(zRnd, 2);
    }
    Vec3 randomPoints(xRnd, yRnd, zRnd);

    Vec3 reflectionDirection = hit.positionHit + normalize(hit.normalVector + randomPoints);
    scattered = Ray(hit.positionHit, reflectionDirection - hit.positionHit);
    return true;
}