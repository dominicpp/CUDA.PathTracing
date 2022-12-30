﻿#include "cuda_runtime.h"
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
#include "material.cuh"

__host__ __device__ Vec3 calculateRadiance(const Ray& ray, Hit* scene, int depth)
{
    RecordHit record;

    if (depth <= 0) { return Vec3(0, 0, 0); }

    if (scene->hitIntersect(ray, 0.001, std::numeric_limits<float>::max(), record))
    {
        // Objects
        //Vec3 target = record.positionHit + record.normalVector + random_in_unit_sphere();
        //return 0.5 * calculateRadiance(Ray(record.positionHit, target - record.positionHit), scene, depth - 1);

        Ray scattered;
        Vec3 attenuation;
        if (depth < 50 && record.material->scatteredRay(ray, record, attenuation, scattered))
        {
            return attenuation * calculateRadiance(scattered, scene, depth + 1);
        }
        else {
            return Vec3(0, 0, 0);
        }


    }
    else {
        // Sky
        Vec3 unit_dir = unit_vector(ray.getDirection());
        float t = 0.5 * (unit_dir.getY() + 1.0);
        return (1.0 - t) * Vec3(1.0, 1.0, 1.0) + t * Vec3(0.5, 0.7, 1.0);
    }
}

int main()
{
    float aspect_ratio = (16 / 8.5);
    int width = 1920;
    int height = static_cast<int>(width / aspect_ratio);
    int sampler = 300;
    float gamma = 2.2f;

    std::ofstream out("doc/img1.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";

    Hit* list[10];
    
    // center sphere
    list[0] = new Sphere(Vec3(0, 0, -0.8), 0.5, new Lambertian(Vec3(0.8, 0.3, 0.3)));
    // behind camera
    list[1] = new Sphere(Vec3(0, 0, 1.5), 0.5, new Lambertian(Vec3(1, 0, 1)));
    list[2] = new Sphere(Vec3(0, 0, -0.25), 0.05, new Metal(Vec3(1, 1, 1), 0));



    list[3] = new Sphere(Vec3(0.75, -0.25, -0.8), 0.3, new Lambertian(Vec3(0.8, 0.8, 0.0)));
    list[4] = new Sphere(Vec3(-0.75, -0.25, -0.8), 0.3, new Lambertian(Vec3(1, 0, 0)));

    // unten
    list[5] = new Sphere(Vec3(0.55, -0.25, -0.3), 0.1, new Metal(Vec3(1, 1, 1), 0));
    list[6] = new Sphere(Vec3(-0.55, -0.25, -0.3), 0.1, new Metal(Vec3(1, 1, 1), 0));

    // oben
    list[7] = new Sphere(Vec3(0.35, 0.2, -0.35), 0.05, new Metal(Vec3(0.6, 0.8, 0.2), 1));
    list[8] = new Sphere(Vec3(-0.35, 0.2, -0.35), 0.05, new Metal(Vec3(0.2, 0.6, 0.8), 1));

    list[9] = new Sphere(Vec3(0, -100.5, -1), 100, new Lambertian(Vec3(1, 1, 1)));




    // list[6] = new Plane(Vec3(0.0, -25.0, 1.0), Vec3(0.0, 0.0, 0.0));

    Hit* scene = new Hitlist(list, 10);

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    Camera* camera = new Camera(viewport_width, viewport_height);
    
    auto a = std::chrono::high_resolution_clock::now();
    for (int y = height - 1; y >= 00; y--)
    {
        std::cerr << "\rRemaining lines scanning: " << y << ' ' << std::flush;
        for (int x = 0; x < width; x++)
        {
            Vec3 imagePixel(0, 0, 0);
            for (int xi = 0; xi < sampler; xi++)
            {
                float u = float(x + random_double()) / float(width);
                float v = float(y + random_double()) / float(height);

                Ray ray = camera->generateRay(u, v);
                imagePixel += calculateRadiance(ray, scene, 4);
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