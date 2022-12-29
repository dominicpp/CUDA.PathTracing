#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>

#include <stdlib.h>

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <limits>

#include "ray.cuh"
#include "hit.cuh"
#include "sphere.cuh"
#include "plane.cuh"
#include "hitlist.cuh"
#include "camera.cuh"

__host__ __device__ Vec3 calculateRadiance(const Ray& r, Hit* world, int depth)
{
    RecordHit record;

    if (depth <= 0) { return Vec3(0, 0, 0); }

    if (world->hitIntersect(r, 0.001, std::numeric_limits<float>::max(), record))
    {
        // Spheres
        Vec3 target = record.positionHit + record.normalVector + random_in_unit_sphere();
        return 0.5 * calculateRadiance(Ray(record.positionHit, target - record.positionHit), world, depth - 1);
    }
    else {
        // Sky
        Vec3 unit_dir = unit_vector(r.getDirection());
        float t = 0.5 * (unit_dir.getY() + 1.0);
        return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
    }
}

int main()
{
    float aspect_ratio = (16 / 8.5);
    int width = 1000;
    int height = static_cast<int>(width / aspect_ratio);
    int sampler = 32;
    float gamma = 2.2f;

    std::ofstream out("out3.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";

    Hit* list[6];
    list[0] = new Sphere(Vec3(0, 0, -0.8), 0.5);
    list[1] = new Sphere(Vec3(0.75, -0.25, -0.8), 0.3);
    list[2] = new Sphere(Vec3(-0.75, -0.25, -0.8), 0.3);
    list[3] = new Sphere(Vec3(1.35, -0.35, -0.9), 0.25);
    list[4] = new Sphere(Vec3(-1.35, -0.35, -0.9), 0.25);
    list[5] = new Sphere(Vec3(0, -100.5, -1), 100);

    Hit* world = new Hitlist(list, 6);

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    Camera* camera = new Camera(viewport_width, viewport_height);
    
    auto a = std::chrono::high_resolution_clock::now();
    for (int y = height - 1; y >= 00; y--)
    {
        std::cerr << "\rScanlines reamining: " << y << ' ' << std::flush;
        for (int x = 0; x < width; x++)
        {
            Vec3 imagePixel(0, 0, 0);
            for (int xi = 0; xi < sampler; xi++)
            {
                float u = float(x + random_double()) / float(width);
                float v = float(y + random_double()) / float(height);

                Ray r = camera->generateRay(u, v);
                imagePixel += calculateRadiance(r, world, 20);
            }
            imagePixel /= float(sampler);

            int r = int(255.9 * (pow(imagePixel[0], 1 / gamma)));
            int g = int(255.9 * (pow(imagePixel[1], 1 / gamma)));
            int b = int(255.9 * (pow(imagePixel[2], 1 / gamma)));
            out << r << " " << g << " " << b << "\n";
        }    
    }
    auto b = std::chrono::high_resolution_clock::now();
    std::cerr << "\n\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(b - a).count() << " seconds\n";
    std::cerr << "\nDone.\n";
}