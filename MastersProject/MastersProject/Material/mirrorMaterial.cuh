#pragma once

#include "../Material/material.cuh"

// Source 1: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, S. Marschner, [Book] “Fundamentals of Computer Graphics,” 3rd ed., 
// A K Peters / CRC Press, 2009, isbn: 9781568814698.
// Source 3: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

class Mirror : public Material
{
    Vec3 m_albedo;

public:
    Mirror() = default;
    Mirror(const Vec3& albedo) : m_albedo(albedo) {}

    virtual Vec3 albedo() const override;
    virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const override;
};

Vec3 Mirror::albedo() const { return m_albedo; }

bool Mirror::scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const
{
    // law of reflection
    Vec3 reflectionDirection = normalize(ray.getDirection()) - 2 * dotProduct(normalize(ray.getDirection()), hit.normalVector) * hit.normalVector;
    scattered = Ray(hit.positionHit, reflectionDirection);
    return true;
}