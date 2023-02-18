#pragma once

#include "../Hit/shape.cuh"

// Source: P. Shirley, “9.1. An Abstract Class for Materials,” in Ray Tracing in One Weekend, 
// [Online] raytracing.github.io, S. Hollasch and T. D. Black, Ed., Available:
// https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 03 January 2023]

struct RecordHit;

class Material
{
public:
    __device__ Material() = default;

    __device__ virtual Vec3 albedo() const = 0;

    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const = 0;
};