#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

Vec3 calculateRadiance(const Ray& ray, Shape* scene, int depth)
{
	RecordHit hit;

	if (depth <= 0) { return Vec3(0.0, 0.0, 0.0); }
	if (scene->hitIntersect(ray, 0.001, std::numeric_limits<float>::max(), hit))
	{
		Ray scattered;
		Vec3 albedo = hit.material->albedo();
		if (hit.material->scatteredRay(ray, hit, scattered))
			return albedo * calculateRadiance(scattered, scene, depth - 1);
		else return Vec3(0.0, 0.0, 0.0);
	}
	else return Vec3(1.0, 1.0, 1.0); // background
}

void raytrace(int width, int height, Camera* camera, Shape* scene, std::ofstream& out, int sampler, float gamma)
{
	for (int y = height-1; y != 0; --y)
	{
		for (int x = 0; x != width; ++x)
		{
			Vec3 imagePixel(0.0, 0.0, 0.0);
			for (int yi = 0; yi < sampler; ++yi)
			{
				for (int xi = 0; xi < sampler; ++xi)
				{
					float ys = float(y + ((yi + random_double())) / sampler) / float(height);
					float xs = float(x + ((xi + random_double())) / sampler) / float(width);

					Ray ray = camera->generateRay(xs, ys);
					imagePixel = imagePixel + calculateRadiance(ray, scene, 15);
				}
			}
			imagePixel = imagePixel / float(sampler * sampler);

			int r = int(255 * (pow(imagePixel[0], 1 / gamma)));
			int g = int(255 * (pow(imagePixel[1], 1 / gamma)));
			int b = int(255 * (pow(imagePixel[2], 1 / gamma)));
			out << r << " " << g << " " << b << "\n";
		}
	}
}

int main()
{
	float aspect_ratio = (16 / 8.5);
	int width = 1920; // resolution
	/*int height = static_cast<int>(width / aspect_ratio);*/
	int height = 960;
	int sampler = 32; // rays per pixel
	float gamma = 2.2f;

	//float viewport_height = 2.0;
	//float viewport_width = aspect_ratio * viewport_height;
	Camera* camera = new Camera(4.0f, 2.0f);

	std::ofstream out("doc/cpp_01.ppm");
	out << "P3\n" << width << " " << height << "\n255\n";
	auto a = std::chrono::high_resolution_clock::now();
	Shape* shapes[13];
	shapes[0] = new Sphere(Vec3(0.0, 0.0, -1.0), 0.5, new Diffuse(Vec3(0.2, 0.6, 0.8))); // center diffuse sphere
	shapes[1] = new Sphere(Vec3(0.0, 0.0, 1.5), 0.5, new Diffuse(Vec3(1.0, 0.0, 1.0))); // behind camera diffuse sphere
	shapes[2] = new Sphere(Vec3(-0.20, -0.45, -0.65), 0.05, new Diffuse(Vec3(1.0, 0.45, 0.5))); // pink diffuse sphere infront of center sphere
	shapes[3] = new Sphere(Vec3(0.78, -0.15, -1.0), 0.3, new PolishedMetal(Vec3(1.0, 1.0, 1.0), 0.23)); // polished metal sphere right from center sphere
	shapes[4] = new Sphere(Vec3(-0.78, -0.15, -1.0), 0.3, new Diffuse(Vec3(1.0, 0.0, 0.0))); // red diffuse sphere
	shapes[5] = new Sphere(Vec3(0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1.0, 1.0, 1.0))); // mirror sphere down right
	shapes[6] = new Sphere(Vec3(-0.75, -0.23, -0.48), 0.1, new Mirror(Vec3(1.0, 1.0, 1.0))); // mirror sphere down left
	shapes[7] = new Sphere(Vec3(0.29, 0.2, -0.39), 0.05, new Diffuse(Vec3(0.2, 0.8, 0.2))); // green sphere up right
	shapes[8] = new Sphere(Vec3(-0.29, 0.2, -0.39), 0.05, new PolishedMetal(Vec3(1.0, 1.0, 1.0), 1.0)); // polished metal sphere up left
	shapes[9] = new Sphere(Vec3(0.0, -100.5, -1.0), 100, new Diffuse(Vec3(0.85, 0.85, 0.85))); // plane sphere
	shapes[10] = new Sphere(Vec3(-0.43, -0.40, -0.85), 0.05, new Mirror(Vec3(1.0, 0.0, 1.0))); // tiny purple mirror sphere 
	shapes[11] = new Sphere(Vec3(0.40, -0.40, -0.75), 0.09, new Mirror(Vec3(1.0, 1.0, 0.0))); // yellow mirror sphere
	shapes[12] = new Sphere(Vec3(-0.15, 0.21, -0.56), 0.06, new Diffuse(Vec3(0.2, 0.8, 0.6))); // aqua sphere on blue sphere
	Shape* scene = new Group(shapes, 13);

	
	raytrace(width, height, camera, scene, out, sampler, gamma);
	auto b = std::chrono::high_resolution_clock::now();
	std::cerr << "\n\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(b - a).count() << " seconds\n";
}