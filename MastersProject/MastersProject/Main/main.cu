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


__global__ void createObjects(Shape** objects, Shape** scene)
{
    objects[0] = new Sphere(Vec3(0.0, 0.0, -1.0), 0.5, new Diffuse(Vec3(0.2, 0.6, 0.8))); // center diffuse sphere
    objects[1] = new Sphere(Vec3(0.0, 0.0, 1.5), 0.5, new Diffuse(Vec3(1.0, 0.0, 1.0))); // behind camera diffuse sphere
    objects[2] = new Sphere(Vec3(-0.20, -0.45, -0.65), 0.05, new Diffuse(Vec3(1.0, 0.45, 0.5))); // pink diffuse sphere infront of center sphere
    objects[3] = new Sphere(Vec3(0.78, -0.15, -1.0), 0.3, new PolishedMetal(Vec3(1.0, 1.0, 1.0), 0.33)); // polished metal sphere right from center sphere
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
}

__global__ void createCamera(Camera** d_camera) { *d_camera = new Camera(4.0f, 2.0f); }

__device__ Vec3 calculateRadiance(const Ray& ray, Shape** scene, int depth, curandStateXORWOW* state)
{
    Ray tempRay = ray;
    Vec3 attenuation = Vec3(0.95f, 0.95f, 1.0f);
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

__device__ void calculatePixelId(int width, int& tidx, int& tidy, int& offsetx, int& offsety, int& gidx, int& gidy) 
{
    tidx = threadIdx.x;
    tidy = threadIdx.y;
    offsetx = blockIdx.x * blockDim.x;
    offsety = blockIdx.y * blockDim.y;
    gidx = tidx + offsetx;
    gidy = tidy + offsety;
}

//##### Setup (R)andom (N)umber (G)enerator #####
__global__ void setupRNG(int width, int height, uint64_t seed, curandStateXORWOW* state)
{
    int tidx, tidy, offsetx, offsety, gidx, gidy;
    calculatePixelId(width, tidx, tidy, offsetx, offsety, gidx, gidy);
    int pixelId = gidy * width + gidx;
    curand_init(seed, pixelId, 0, &state[pixelId]);
}

__global__ void raytracing(Vec3* buffer, int width, int height, Camera** camera, Shape** scene, curandStateXORWOW* state, int sample, float gamma)
{
    int tidx, tidy, offsetx, offsety, gidx, gidy;
    calculatePixelId(width, tidx, tidy, offsetx, offsety, gidx, gidy);
    int pixelId = gidy * width + gidx;
    curandStateXORWOW tempState = state[pixelId];
    
    //##### Stratified Sampling #####
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
    buffer[pixelId] = setPixel;
    state[pixelId] = tempState;
}



int main()
{
    // resolution in x & y dimension / number of threads for each dimension
    int nx = 1920;
    int ny = 960;
    // number of (thread-)blocks in x & y dimension
    int tx = 32;
    int ty = 32;
    int sample = 10; // rays per pixel -> in fact 32x32 with Stratified Sampling
    float gamma = 2.2f; // corrected gamma value

    int allPixels = nx * ny;

    std::ofstream out("doc/cuda_test02.ppm");
    std::cerr << "Rendering a " << nx << "x" << ny << " image with " << sample * sample << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //##### CUDA MEMORY ALLOCATION #####
    curandStateXORWOW* d_state; // Random Number Generator
    cudaMallocManaged((void**)&d_state, allPixels* sizeof(curandStateXORWOW));
    Vec3* d_buffer;
    cudaMallocManaged((void**)&d_buffer, allPixels * sizeof(Vec3));
    Shape** d_objects;
    cudaMallocManaged((void**)&d_objects, 13 * sizeof(Shape*));
    Shape** d_scene;
    cudaMallocManaged((void**)&d_scene, sizeof(Shape*));
    Camera** d_camera;
    cudaMallocManaged((void**)&d_camera, sizeof(Camera*));

 
    // how many threads per block in x and y dimension
    // 1024 threads per block is max!
    dim3 block(18, 6, 1);
    // how many thread blocks in x and y dimension
    dim3 grid(107, 160, 1);

    auto start = std::chrono::high_resolution_clock::now();
    //##### KERNEL 1 #####
    createObjects << <1, 1, 0, 0 >> > (d_objects, d_scene);
    cudaGetLastError();
    cudaDeviceSynchronize();

    //##### KERNEL 2 #####
    createCamera << <1, 1, 0, 0 >> > (d_camera);
    cudaGetLastError();
    cudaDeviceSynchronize();
    
    //##### KERNEL 3 #####
    uint64_t d_seed = 2023;
    setupRNG << <grid, block, 0, 0 >> > (nx, ny, d_seed, d_state);
    cudaGetLastError();
    cudaDeviceSynchronize();

    //##### KERNEL 4 #####
    raytracing << <grid, block, 0, 0 >> > (d_buffer, nx, ny, d_camera, d_scene, d_state, sample, gamma);
    cudaGetLastError();
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds\n";
    out << "P3\n" << nx << " " << ny << "\n255\n";
    
    for (int y = ny - 1; y != 0; --y)
    {
        for (int x = 0; x != nx; ++x)
        {
            int pixelId = y * nx + x;
            int r = int(255 * d_buffer[pixelId][0]);
            int g = int(255 * d_buffer[pixelId][1]);
            int b = int(255 * d_buffer[pixelId][2]);
            out << r << " " << g << " " << b << "\n";
        }
    }

    //##### free memory on gpu #####
    cudaFree(d_state);
    cudaFree(d_buffer);
    cudaFree(d_objects);
    cudaFree(d_scene);
    cudaFree(d_camera);
    //##### remove all gpu allocations #####
    cudaDeviceReset();
}