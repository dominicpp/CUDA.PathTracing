#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>

#include <fstream>
#include <iostream>
#include <chrono>

#include "../Utils/vec3.cuh"
#include "../Ray/ray.cuh"
#include "../Hit/group.cuh"
#include "../Camera/camera.cuh"
#include "../Material/material.cuh"
#include "../Material/diffuseMaterial.cuh"
#include "../Material/mirrorMaterial.cuh"
#include "../Material/polishedMetalMaterial.cuh"
#include "../Objects/sphere.cuh"


__device__ Vec3 calculateRadiance(const Ray& ray, Shape** scene, int depth, curandStateXORWOW* state)
{
    Ray tempRay = ray;
    Vec3 attenuation = Vec3(1.0f, 1.0f, 1.0f);
    int bounces = 0;

    while (bounces < depth)
    {
        RecordHit hit;
        if ((*scene)->hitIntersect(tempRay, 0.001f, FLT_MAX, hit))
        {
            Ray scattered;
            Vec3 albedo = hit.material->albedo();
            if (hit.material->scatteredRay(tempRay, hit, scattered, state))
            {
                attenuation = attenuation * albedo;
                tempRay = scattered;
                bounces++;
                continue;
            }
            else return Vec3(0.0f, 0.0f, 0.0f);
        }
        // background
        return attenuation;
    }
    return Vec3(0.0f, 0.0f, 0.0f);
}

//#############################################
__global__ void render_init(int width, int height, curandStateXORWOW* state)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int offsetx = blockIdx.x * blockDim.x;
    int offsety = blockIdx.y * blockDim.y;
    int gidx = tidx + offsetx;
    int gidy = tidy + offsety;
    if ((gidx >= width) || (gidy >= height)) return;
    int pixelIndex = gidy * width + gidx;
    curand_init(2023, pixelIndex, 0, &state[pixelIndex]);
}
//#############################################

__global__ void raytrace(Vec3* buffer, int width, int height, Camera** camera, Shape** scene, curandStateXORWOW* state, int sample, float gamma)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int offsetx = blockIdx.x * blockDim.x;
    int offsety = blockIdx.y * blockDim.y;
    int gidx = tidx + offsetx;
    int gidy = tidy + offsety;
    if ((gidx >= width) || (gidy >= height)) return;
    int pixelIndex = gidy * width + gidx;

    curandStateXORWOW tempState = state[pixelIndex];
    Vec3 color(0, 0, 0);
    for (int x = 0; x < sample; ++x)
    {
        for (int y = 0; y < sample; ++y)
        {
            float rx = curand_uniform(&tempState);
            float ry = curand_uniform(&tempState);
            float sx = (gidx + (x + rx) / sample) / float(width);
            float sy = (gidy + (y + ry) / sample) / float(height);
            Ray ray = (*camera)->generateRay(sx, sy);
            color = color + calculateRadiance(ray, scene, 15, &tempState);
        }
    }

    Vec3 setPixel = color;
    setPixel = color / float(sample * sample);
    setPixel[0] = pow(setPixel[0], 1 / gamma);
    setPixel[1] = pow(setPixel[1], 1 / gamma);
    setPixel[2] = pow(setPixel[2], 1 / gamma);
    buffer[pixelIndex] = setPixel;
    state[pixelIndex] = tempState;
}

__global__ void create_world(Shape** objects, Shape** scene, Camera** d_camera)
{
    objects[0] = new Sphere(Vec3(0.0, 0.0, -1.0), 0.5, new Diffuse(Vec3(0.2, 0.6, 0.8))); // center diffuse sphere
    objects[1] = new Sphere(Vec3(0.0, 0.0, 1.5), 0.5, new Diffuse(Vec3(1.0, 0.0, 1.0))); // behind camera diffuse sphere
    objects[2] = new Sphere(Vec3(-0.20, -0.45, -0.65), 0.05, new Diffuse(Vec3(1.0, 0.45, 0.5))); // pink diffuse sphere infront of center sphere
    objects[3] = new Sphere(Vec3(0.78, -0.15, -1.0), 0.3, new PolishedMetal(Vec3(1.0, 1.0, 1.0), 0.23)); // polished metal sphere right from center sphere
    objects[4] = new Sphere(Vec3(-0.78, -0.15, -1.0), 0.3, new Diffuse(Vec3(1.0, 0.0, 0.0))); // red diffuse sphere
    objects[5] = new Sphere(Vec3(0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1.0, 1.0, 1.0))); // mirror sphere down right
    objects[6] = new Sphere(Vec3(-0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1.0, 1.0, 1.0))); // mirror sphere down left
    objects[7] = new Sphere(Vec3(0.29, 0.2, -0.39), 0.05, new Diffuse(Vec3(0.2, 0.8, 0.2))); // green sphere up right
    objects[8] = new Sphere(Vec3(-0.29, 0.2, -0.39), 0.05, new PolishedMetal(Vec3(1.0, 1.0, 1.0), 1.0)); // polished metal sphere up left
    objects[9] = new Sphere(Vec3(0.0, -100.5, -1.0), 100, new Diffuse(Vec3(0.85, 0.85, 0.85))); // plane sphere
    objects[10] = new Sphere(Vec3(-0.43, -0.40, -0.85), 0.05, new Mirror(Vec3(1.0, 0.0, 1.0))); // tiny purple mirror sphere 
    objects[11] = new Sphere(Vec3(0.40, -0.40, -0.75), 0.09, new Mirror(Vec3(1.0, 1.0, 0.0))); // yellow mirror sphere
    objects[12] = new Sphere(Vec3(-0.15, 0.21, -0.56), 0.06, new Diffuse(Vec3(0.2, 0.8, 0.6))); // aqua sphere on blue sphere

    *scene = new Group(objects, 13);
    *d_camera = new Camera(4.0f, 2.0f);
}

int main()
{
    // resolution in x & y dimension / number of threads for each dimension
    int nx = 1200;
    int ny = 600;
    // number of (thread-)blocks in x & y dimension
    int tx = 32;
    int ty = 32;
    int sample = 10; // rays per pixel -> in fact 32x32 with Stratified Sampling
    float gamma = 2.2f; // corrected gamma value

    int allPixels = nx * ny;
    float bufferSize = allPixels * sizeof(Vec3);

    std::ofstream out("doc/cuda_test.ppm");
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << sample * sample << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    // ########## CUDA MEMORY ALLOCATION
    curandStateXORWOW* d_state; // Random Number Generator
    cudaMallocManaged((void**)&d_state, allPixels * sizeof(curandStateXORWOW));
    Vec3* d_buffer;
    cudaMallocManaged((void**)&d_buffer, bufferSize);
    Shape** d_objects;
    cudaMallocManaged((void**)&d_objects, 12 * sizeof(Shape*));
    Shape** d_scene;
    cudaMallocManaged((void**)&d_scene, sizeof(Shape*));
    Camera** d_camera;
    cudaMallocManaged((void**)&d_camera, sizeof(Camera*));
    // ##########

    dim3 grid(nx / tx + 1, ny / ty + 1, 1);
    dim3 block(tx, ty, 1);

    auto a = std::chrono::high_resolution_clock::now();
    // KERNEL 1
    create_world << <1, 1 >> > (d_objects, d_scene, d_camera);
    cudaGetLastError();
    cudaDeviceSynchronize();
    //#############################################
    // KERNEL 2
    render_init << <grid, block >> > (nx, ny, d_state);
    cudaGetLastError();
    cudaDeviceSynchronize();
    //#############################################
    // KERNEL 3
    raytrace << <grid, block >> > (d_buffer, nx, ny, d_camera, d_scene, d_state, sample, gamma);
    cudaGetLastError();
    cudaDeviceSynchronize();
    auto b = std::chrono::high_resolution_clock::now();
    
    std::cerr << "\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(b - a).count() << " seconds\n";
    out << "P3\n" << nx << " " << ny << "\n255\n";
    for (int y = ny; y != 0; --y)
    {
        for (int x = 0; x != nx; ++x)
        {
            int pixelIndex = y * nx + x;
            int r = int(255.99 * d_buffer[pixelIndex][0]);
            int g = int(255.99 * d_buffer[pixelIndex][1]);
            int b = int(255.99 * d_buffer[pixelIndex][2]);
            out << r << " " << g << " " << b << "\n";
        }
    }

    // free memory on device
    cudaFree(d_camera);
    cudaFree(d_scene);
    cudaFree(d_objects);
    cudaFree(d_state);
    cudaFree(d_buffer);
    // remove all device allocations
    cudaDeviceReset();
}