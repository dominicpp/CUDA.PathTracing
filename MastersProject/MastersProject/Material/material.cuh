#pragma once

#include "../Hit/shape.cuh"

struct RecordHit;

class Material
{
public:
    __device__ Material() = default;

    __device__ virtual Vec3 albedo() const = 0;

    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const = 0;
};