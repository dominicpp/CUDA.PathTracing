#pragma once

#include "../Hit/shape.cuh"

// Source 1: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

struct RecordHit;

class Material
{
public:
    __device__ Material() = default;

    __device__ virtual Vec3 albedo() const = 0;

    __device__ virtual bool scatteredRay(const Ray& ray, const RecordHit& hit, Ray& scattered, curandStateXORWOW* state) const = 0;
};