#pragma once

#include "../Hit/shape.cuh"

struct RecordHit;

class Material
{
public:
    Material() = default;

    virtual Vec3 albedo() const = 0;

    virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered) const = 0;
};