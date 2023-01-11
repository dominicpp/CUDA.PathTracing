#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>

#include <stdlib.h>

#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <limits>
#include <vector>

#include "../Ray/ray.cuh"
#include "../Hit/hit.cuh"
#include "../Objects/sphere.cuh"
#include "../Objects/plane.cuh"
#include "../Hit/group.cuh"
#include "../Camera/camera.cuh"
#include "../Material/diffuse.cuh"
#include "../Material/polishedMetal.cuh"
#include "../Material/mirror.cuh"

// Recursion -> Ray bouncing around / Path Tracing
__host__ __device__ Vec3 calculateRadiance(const Ray& ray, Hit* scene, int depth)
{
    RecordHit record;

    if (depth <= 0) { return Vec3(0, 0, 0); }

    if (scene->hitIntersect(ray, 0.001, std::numeric_limits<float>::max(), record))
    {
        // Spheres
        Ray scattered;
        Vec3 weakening;
        if (depth <= 50 && record.material->scatteredRay(ray, record, weakening, scattered))
            return weakening * calculateRadiance(scattered, scene, depth - 1); // Tramberend/Diffuse Reflexion Video Minute 6:15
        else return Vec3(0.0, 0.0, 0.0);
    }
    else 
        return Vec3(1.0, 1.0, 1.0); // Background
}

int main()
{
    float aspect_ratio = (16 / 8.5);
    int width = 800;
    int height = static_cast<int>(width / aspect_ratio);
    int sampler = 40; // rays per pixel
    float gamma = 2.2f;

    std::ofstream out("doc/test11.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";

    std::vector<Hit*> shapes;
    shapes.push_back(new Sphere(Vec3(0, 0, -1.0), 0.5, new Diffuse(Vec3(0.8, 0.3, 0.3)))); // center sphere
    shapes.push_back(new Sphere(Vec3(0, 0, 1.5), 0.5, new Diffuse(Vec3(1, 0, 1)))); // behind camera
    shapes.push_back(new Sphere(Vec3(0, 0, -0.25), 0.05, new PolishedMetal(Vec3(1, 1, 1), 0))); // glass sphere infront of camera
    shapes.push_back(new Sphere(Vec3(0.78, -0.15, -1), 0.3, new PolishedMetal(Vec3(0.2, 0.6, 0.8), 1))); // blue metal sphere
    shapes.push_back(new Sphere(Vec3(-0.78, -0.15, -1), 0.3, new Diffuse(Vec3(1, 0, 0)))); // red diffuse sphere
    shapes.push_back(new Sphere(Vec3(0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1, 1, 1)))); // sphere down right
    shapes.push_back(new Sphere(Vec3(-0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1, 1, 1)))); // sphere down left
    shapes.push_back(new Sphere(Vec3(0.29, 0.2, -0.39), 0.05, new PolishedMetal(Vec3(0.6, 0.8, 0.2), 1))); // sphere up right
    shapes.push_back(new Sphere(Vec3(-0.29, 0.2, -0.39), 0.05, new Diffuse(Vec3(0.8, 0.3, 0.1)))); // sphere up left
    shapes.push_back(new Sphere(Vec3(0, -100.5, -1), 100, new Diffuse(Vec3(0.85, 0.85, 0.85)))); // plane sphere
    Hit* scene = new Group(shapes);

    auto viewport_height = 2.0;
    auto viewport_width = aspect_ratio * viewport_height;
    Camera* camera = new Camera(viewport_width, viewport_height);
    
    auto a = std::chrono::high_resolution_clock::now();
    for (int y = height - 1; y >= 00; y--)
    {
        std::cerr << "\r##### Remaining lines scanning: " << y << ' ' << std::flush;
        for (int x = 0; x < width; x++)
        {
            Vec3 imagePixel(0, 0, 0);
            for (int xi = 0; xi < sampler; xi++)
            {
                float u = float(x + random_double()) / float(width);
                float v = float(y + random_double()) / float(height);

                Ray ray = camera->generateRay(u, v);
                imagePixel += calculateRadiance(ray, scene, 20);
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