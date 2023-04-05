#pragma once

#include "../Material/material.cuh"

// Source 1: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, S. Marschner, [Book] “Fundamentals of Computer Graphics,” 3rd ed., 
// A K Peters / CRC Press, 2009, isbn: 9781568814698.
// Source 3: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

class Diffuse : public Material
{
    Vec3 m_albedo;

public:
    Diffuse() = default;
    Diffuse(const Vec3& albedo) : m_albedo(albedo) {}

    virtual Vec3 albedo() const override;
    virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const override;
};

Vec3 Diffuse::albedo() const { return m_albedo; }

bool Diffuse::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const
{
    // Monte Carlo Rejection Method
    double sum, xRnd, yRnd, zRnd;
    sum = xRnd = yRnd = zRnd = 1.0;
    while (sum >= 1.0)
    {
        xRnd = random_number() * 2 - 1.0;
        yRnd = random_number() * 2 - 1.0;
        zRnd = random_number() * 2 - 1.0;
        sum = pow(xRnd, 2) + pow(yRnd, 2) + pow(zRnd, 2);
    }
    Vec3 randomPoints(xRnd, yRnd, zRnd);

    Vec3 reflectionDirection = hit.positionHit + normalize(hit.normalVector + randomPoints);
    scattered = Ray(hit.positionHit, reflectionDirection - hit.positionHit);
    return true;
}