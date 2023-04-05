#pragma once

#include <cmath>
#include <random>

// Source: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.

struct Vec3
{
    float a[3];

     Vec3() = default;
     Vec3(float xR, float yG, float zB);
     inline float operator[](int i) const { return a[i]; }
     inline float& operator[](int i) { return a[i]; }
     inline const Vec3& operator+() const { return *this; }
     inline Vec3 operator-() const { return Vec3(-a[0], -a[1], -a[2]); }
     friend Vec3 operator+(const Vec3& v1, const Vec3& v2);
     friend Vec3 operator-(const Vec3& v1, const Vec3& v2);
     friend Vec3 operator*(const Vec3& v1, const Vec3& v2);
     friend Vec3 operator*(float scalar, const Vec3& v);
     friend Vec3 operator/(const Vec3& v, const float scalar);
     friend std::ostream& operator<<(std::ostream& os, const Vec3& t);
     float length() const;
     float length(Vec3& a) const;
     float squared_length() const;
     float squared_length(Vec3 a) const;
};

inline Vec3::Vec3(float xR, float yG, float zB)
{
    a[0] = xR; a[1] = yG; a[2] = zB;
}

inline Vec3 operator+(const Vec3& v1, const Vec3& v2)
{
    return Vec3(v1.a[0] + v2.a[0], v1.a[1] + v2.a[1], v1.a[2] + v2.a[2]);
}

inline Vec3 operator-(const Vec3& v1, const Vec3& v2)
{
    return Vec3(v1.a[0] - v2.a[0], v1.a[1] - v2.a[1], v1.a[2] - v2.a[2]);
}

inline Vec3 operator*(const Vec3& v1, const Vec3& v2)
{
    return Vec3(v1.a[0] * v2.a[0], v1.a[1] * v2.a[1], v1.a[2] * v2.a[2]);
}

inline Vec3 operator*(float scalar, const Vec3& v)
{
    return Vec3(scalar * v.a[0], scalar * v.a[1], scalar * v.a[2]);
}

inline Vec3 operator/(const Vec3& v, const float scalar)
{
    return Vec3(v.a[0] / scalar, v.a[1] / scalar, v.a[2] / scalar);
}

inline std::ostream& operator<<(std::ostream& os, const Vec3& t)
{
    os << "(" << t.a[0] << ", " << t.a[1] << ", " << t.a[2] << ")";
    return os;
}

inline float Vec3::length() const
{
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

inline float Vec3::length(Vec3& a) const
{
    return a.length();
}

inline float Vec3::squared_length() const
{
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

inline float Vec3::squared_length(Vec3 a) const
{
    return a.squared_length();
}

inline float dotProduct(const Vec3& a, const Vec3& b)
{
    return a.a[0] * b.a[0] + a.a[1] * b.a[1] + a.a[2] * b.a[2];
}

inline Vec3 normalize(Vec3 v)
{
    return v / v.length();
}

inline double random_number() 
{
    return static_cast<double>(std::rand()) / static_cast<double>(RAND_MAX);
}