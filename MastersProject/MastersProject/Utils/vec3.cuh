#pragma once

#include <cmath>
#include <random>

struct Vec3 {
    float a[3];

    __host__ __device__ Vec3() = default;
    __host__ __device__ Vec3(float xR, float yG, float zB) { a[0] = xR; a[1] = yG; a[2] = zB; }

    __host__ __device__ inline float getX() const { return a[0]; }
    __host__ __device__ inline float getY() const { return a[1]; }
    __host__ __device__ inline float getZ() const { return a[2]; }

    __host__ __device__ inline float getR() const { return a[0]; }
    __host__ __device__ inline float getG() const { return a[1]; }
    __host__ __device__ inline float getB() const { return a[2]; }

    __host__ __device__ inline const Vec3& operator+() const { return *this; }
    __host__ __device__ inline Vec3 operator-() const { return Vec3(-a[0], -a[1], -a[2]); }
    
    __host__ __device__ inline float operator[](int i) const { return a[i]; }
    __host__ __device__ inline float& operator[](int i) { return a[i]; }

    __host__ __device__ inline Vec3& operator+=(const Vec3& v2);
    __host__ __device__ inline Vec3& operator/=(const float t);

    __host__ __device__ inline float length() const 
    {
        return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
    }

    __host__ __device__ inline float length(Vec3& a) const
    {
        return a.length();
    }

    __host__ __device__ inline float squared_length() const 
    {
        return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    }

    __host__ __device__ inline float squared_length(Vec3 a) const
    {
        return a.squared_length();
    }
};

__host__ __device__ inline std::ostream& operator<<(std::ostream& os, const Vec3& t) 
{
    os << "(" << t[0] << ", " << t[1] << ", " << t[2] << ")";
    return os;
}

__host__ __device__ inline Vec3 operator+(const Vec3& v1, const Vec3& v2) 
{
    return Vec3(v1.a[0] + v2.a[0], v1.a[1] + v2.a[1], v1.a[2] + v2.a[2]);
}

__host__ __device__ inline Vec3 operator-(const Vec3& v1, const Vec3& v2) 
{
    return Vec3(v1.a[0] - v2.a[0], v1.a[1] - v2.a[1], v1.a[2] - v2.a[2]);
}

__host__ __device__ inline Vec3 operator*(const Vec3& v1, const Vec3& v2) {
    return Vec3(v1.a[0] * v2.a[0], v1.a[1] * v2.a[1], v1.a[2] * v2.a[2]);
}

__host__ __device__ inline Vec3 operator*(float t, const Vec3& v) {
    return Vec3(t * v.a[0], t * v.a[1], t * v.a[2]);
}

__host__ __device__ inline Vec3 operator/(const Vec3& v, const float t) {
    return Vec3(v.a[0] / t, v.a[1] / t, v.a[2] / t);
}

__host__ __device__ inline Vec3& Vec3::operator+=(const Vec3& v) {
    a[0] += v.a[0];
    a[1] += v.a[1];
    a[2] += v.a[2];
    return *this;
}

__host__ __device__ inline Vec3& Vec3::operator/=(const float t) {
    float k = 1.0 / t;

    a[0] *= k;
    a[1] *= k;
    a[2] *= k;
    return *this;
}

__host__ __device__ inline float dotProduct(const Vec3& a, const Vec3& b) {
    return a.a[0] * b.a[0] + a.a[1] * b.a[1] + a.a[2] * b.a[2];
}

__host__ __device__ inline Vec3 normalize(Vec3 v) {
    return v / v.length();
}

__host__ __device__ inline double random_double() {
    double min = 0.0;
    double max = 1.0;
    double random = ((double)rand()) / RAND_MAX;
    double range = (max - min) * random;
    double number = min + range;
    return number;
}

// Colors
Vec3 c_red = Vec3(1, 0, 0);
Vec3 c_turquoise = Vec3(0.2, 0.6, 0.8);
Vec3 c_purple = Vec3(1, 0, 1);
Vec3 c_reflection = Vec3(1, 1, 1);