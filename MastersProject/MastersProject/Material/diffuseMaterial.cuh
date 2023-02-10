#pragma once

#include "../Material/material.cuh"

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
    // Monte Carlo
    double sum, xRnd, yRnd, zRnd;
    do {
        xRnd = random_double() * 2 - 1.0;
        yRnd = random_double() * 2 - 1.0;
        zRnd = random_double() * 2 - 1.0;
        sum = pow(xRnd, 2) + pow(yRnd, 2) + pow(zRnd, 2);
    } while (sum >= 1.0);
    Vec3 randomPoints(xRnd, yRnd, zRnd);

    Vec3 reflectionDirection = hit.positionHit + normalize(hit.normalVector + randomPoints);
    scattered = Ray(hit.positionHit, reflectionDirection - hit.positionHit);
    return true;
}