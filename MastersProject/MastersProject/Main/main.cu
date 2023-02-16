﻿#include "cuda_runtime.h"
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


#define W 1920
#define H 960
#define SAMPLES 32 // Stratified Sampling 32*32=1024 samples
#define GAMMA 2.2f

__global__ void createObjects(Shape** objects, Shape** scene)
{
    int count = 0;
    objects[count++] = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new Diffuse(Vec3(0.2f, 0.6f, 0.8f))); // center diffuse sphere
    objects[count++] = new Sphere(Vec3(0.0f, 0.0f, 1.5f), 0.5f, new Diffuse(Vec3(1.0f, 0.0f, 1.0f))); // behind camera diffuse sphere
    objects[count++] = new Sphere(Vec3(-0.20f, -0.45f, -0.65f), 0.05f, new Diffuse(Vec3(1.0f, 0.45f, 0.5f))); // pink diffuse sphere infront of center sphere
    objects[count++] = new Sphere(Vec3(0.78f, -0.15f, -1.0f), 0.3f, new PolishedMetal(Vec3(1.0f, 1.0f, 1.0f), 0.33f)); // polished metal sphere right from center sphere
    objects[count++] = new Sphere(Vec3(-0.78f, -0.15f, -1.0f), 0.3f, new Diffuse(Vec3(1.0f, 0.0f, 0.0f))); // red diffuse sphere
    objects[count++] = new Sphere(Vec3(0.75f, -0.23f, -0.48f), 0.1f, new Mirror(Vec3(1.0f, 1.0f, 1.0f))); // mirror sphere down right
    objects[count++] = new Sphere(Vec3(-0.75f, -0.23f, -0.48f), 0.1f, new Mirror(Vec3(1.0f, 1.0f, 1.0f))); // mirror sphere down left
    objects[count++] = new Sphere(Vec3(0.29f, 0.2f, -0.39f), 0.05f, new Diffuse(Vec3(0.2f, 0.8f, 0.2f))); // green sphere up right
    objects[count++] = new Sphere(Vec3(-0.29f, 0.2f, -0.39f), 0.05f, new PolishedMetal(Vec3(1.0f, 1.0f, 1.0f), 1.0f)); // polished metal sphere up left
    objects[count++] = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f, new Diffuse(Vec3(0.85f, 0.85f, 0.85f))); // plane sphere
    objects[count++] = new Sphere(Vec3(-0.43f, -0.40f, -0.85f), 0.05f, new Mirror(Vec3(1.0f, 0.0f, 1.0f))); // tiny purple mirror sphere 
    objects[count++] = new Sphere(Vec3(0.40f, -0.40f, -0.75f), 0.09f, new Mirror(Vec3(1.0f, 1.0f, 0.0f))); // yellow mirror sphere
    objects[count++] = new Sphere(Vec3(-0.15f, 0.21f, -0.56f), 0.06f, new Diffuse(Vec3(0.2f, 0.8f, 0.6f))); // aqua sphere on blue sphere
    *scene = new Group(objects, count);
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
    if((gidx >= width) || (gidy >= height)) return;
    curand_init(seed, pixelId, 0, &state[pixelId]);
}

__global__ void raytracing(Vec3* buffer, int width, int height, Camera** camera, Shape** scene, curandStateXORWOW* state, int sample, float gamma)
{
    int tidx, tidy, offsetx, offsety, gidx, gidy;
    calculatePixelId(width, tidx, tidy, offsetx, offsety, gidx, gidy);
    int pixelId = gidy * width + gidx;
    if ((gidx >= width) || (gidy >= height)) return;
    curandStateXORWOW tempState = state[pixelId];
    
    //##### Stratified Sampling #####
    Vec3 color(0.0f, 0.0f, 0.0f);
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
    // number of threads per block in x and y dimension
    int tx = 16;
    int ty = 16;
    int allPixels = W * H;

    std::ofstream out("doc/cuda_01.ppm");
    std::cerr << "Rendering a " << W << "x" << H << " image with " << SAMPLES * SAMPLES << " samples per pixel ";
    std::cerr << "in " << tx << "x" << ty << " blocks.\n";

    //##### CUDA MEMORY ALLOCATION #####
    curandStateXORWOW* d_state; // Random Number Generator
    cudaMallocManaged((void**)&d_state, allPixels * sizeof(curandStateXORWOW));
    Vec3* d_buffer;
    cudaMallocManaged((void**)&d_buffer, allPixels * sizeof(Vec3));
    Shape** d_objects;
    cudaMallocManaged((void**)&d_objects, 13 * sizeof(Shape*));
    Shape** d_scene;
    cudaMallocManaged((void**)&d_scene, sizeof(Shape*));
    Camera** d_camera;
    cudaMallocManaged((void**)&d_camera, sizeof(Camera*));

    //##### CUDA KERNEL ARGS #####
    dim3 block(tx, ty, 1); // how many threads per block in x and y dimension -> 32x32 = 1024 threads
    dim3 grid(W / block.x, H / block.y, 1); // how many thread blocks in x and y dimension

    auto start = std::chrono::high_resolution_clock::now();
    //##### KERNEL 1 #####
    {
        createObjects << <1, 1, 0, 0 >> > (d_objects, d_scene);
		cudaGetLastError();
		cudaDeviceSynchronize();
    }

    //##### KERNEL 2 #####
    {
        createCamera << <1, 1, 0, 0 >> > (d_camera);
        cudaGetLastError();
        cudaDeviceSynchronize();
    }
    
    //##### KERNEL 3 #####
    {
        uint64_t d_seed = 2023;
        setupRNG << <grid, block, 0, 0 >> > (W, H, d_seed, d_state);
        cudaGetLastError();
        cudaDeviceSynchronize();
    }

    //##### KERNEL 4 #####
    {
        raytracing << <grid, block, 0, 0 >> > (d_buffer, W, H, d_camera, d_scene, d_state, SAMPLES, GAMMA);
        cudaGetLastError();
        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cerr << "\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds\n";
    out << "P3\n" << W << " " << H << "\n255\n";
    
    for (int y = H - 1; y != 0; --y)
    {
        for (int x = 0; x != W; ++x)
        {
            int pixelId = y * W + x;
            int r = 255 * d_buffer[pixelId][0];
            int g = 255 * d_buffer[pixelId][1];
            int b = 255 * d_buffer[pixelId][2];
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