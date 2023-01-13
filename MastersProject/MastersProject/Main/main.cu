﻿#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include <iostream>
#include <chrono>

#include "../Ray/ray.cuh"
#include "../Camera/camera.cuh"
#include "../Hit/hit.cuh"
#include "../Hit/group.cuh"
#include "../Objects/sphere.cuh"
#include "../Objects/plane.cuh"
#include "../Material/diffuse.cuh"
#include "../Material/polishedmetal.cuh"
#include "../Material/mirror.cuh"

// Recursion -> Ray bouncing around / Path Tracing
__host__ __device__ Vec3 calculateRadiance(const Ray& ray, Hit* scene, int depth)
{
    RecordHit hit;

    if (depth <= 0) { return Vec3(0, 0, 0); }
    if (scene->hitIntersect(ray, 0.001, std::numeric_limits<float>::max(), hit))
    {
        // Spheres
        Ray scattered;
        Vec3 weakening;
        if (/*depth <= 50 && */hit.material->scatteredRay(ray, hit, weakening, scattered))
            return weakening * calculateRadiance(scattered, scene, depth - 1); // Tramberend/Diffuse Reflexion Video Minute 6:15
        else return Vec3(0.0, 0.0, 0.0);
    }
    else 
        return Vec3(1.0, 1.0, 1.0); // background
}

int main()
{
    float aspect_ratio = (16 / 8.5);
    int width = 600; // 2k
    int height = static_cast<int>(width / aspect_ratio);
    int sampler = 80; // rays per pixel
    float gamma = 2.2f;

    std::ofstream out("doc/test5_polishedMetal_altered6_sf05.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";

    Hit* shapes[13];
    shapes[0] = new Sphere(Vec3(0, 0, -1.0), 0.5, new Diffuse(c_turquoise)); // center sphere
    shapes[1] = new Sphere(Vec3(0, 0, 1.5), 0.5, new Diffuse(c_purple)); // behind camera
    shapes[2] = new Sphere(Vec3(-0.20, -0.45, -0.65), 0.05, new Diffuse(Vec3(1.0, 0.45, 0.5))); // glass sphere infront of camera
    shapes[3] = new Sphere(Vec3(0.78, -0.15, -1), 0.3, new PolishedMetal(c_reflection, 0)); //################################ bluish metal sphere
    shapes[4] = new Sphere(Vec3(-0.78, -0.15, -1), 0.3, new Diffuse(c_red)); // red diffuse sphere
    shapes[5] = new Sphere(Vec3(0.75, -0.23, -0.48), 0.1, new Mirror(c_reflection)); // mirror sphere down right
    shapes[6] = new Sphere(Vec3(-0.75, -0.23, -0.48), 0.1, new Mirror(c_reflection)); // mirror sphere down left
    shapes[7] = new Sphere(Vec3(0.29, 0.2, -0.39), 0.05, new Diffuse(Vec3(0.2, 0.8, 0.2))); // sphere up right
    shapes[8] = new Sphere(Vec3(-0.29, 0.2, -0.39), 0.05, new PolishedMetal(c_reflection, 1.0)); // sphere up left
    shapes[9] = new Sphere(Vec3(0, -100.5, -1), 100, new Diffuse(Vec3(0.85, 0.85, 0.85))); // plane sphere
    shapes[10] = new Sphere(Vec3(-0.43, -0.40, -0.85), 0.05, new Mirror(c_purple)); 
    shapes[11] = new Sphere(Vec3(0.40, -0.40, -0.75), 0.09, new Mirror(Vec3(1.0, 1.0, 0)));
    shapes[12] = new Sphere(Vec3(-0.15, 0.21, -0.56), 0.06, new Diffuse(Vec3(0.2,0.8,0.6)));
    Hit* scene = new Group(shapes, 13);

    float viewport_height = 2.0;
    float viewport_width = aspect_ratio * viewport_height;
    Camera* camera = new Camera(viewport_width, viewport_height);
    
    auto a = std::chrono::high_resolution_clock::now();
    for (int y = height - 1; y >= 00; --y)
    {
        std::cerr << "\r##### Remaining lines scanning: " << y << ' ' << std::flush;
        for (int x = 0; x < width; ++x)
        {
            Vec3 imagePixel(0, 0, 0);
            for (int n = 0; n < sampler; ++n)
            {
                float xs = float(x + random_double()) / float(width);
                float ys = float(y + random_double()) / float(height);

                Ray ray = camera->generateRay(xs, ys);
                imagePixel += calculateRadiance(ray, scene, 8);
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