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

// Source 3: P. Shirley, R. K. Morley, [Book] “Realistic Ray Tracing,” 2nd ed., 
// Routledge, 2008, isbn: 9781568814612.
// Source 2: P. Shirley, [eBook] “Ray Tracing in One Weekend, ” vers. 3.2.3, S. Hollaschand and T.D. Black, Ed., Peter Shirley,
// 2018 - 2020, Available: https://raytracing.github.io/books/RayTracingInOneWeekend.html [Accessed 19 November 2022].

#define W 1920
#define H 960
#define SAMPLES 32 // Stratified Sampling 32*32=1024 samples
#define GAMMA 2.2f

// recursive algorithm to compute the radiance of a given ray
Vec3 calculateRadiance(const Ray& ray, Shape* scene, int depth)
{
	RecordHit hit;

	if (depth <= 0) { return Vec3(0.0f, 0.0f, 0.0f); }
	if (scene->hitIntersect(ray, 0.001f, std::numeric_limits<float>::max(), hit))
	{
		Ray scattered;
		Vec3 albedo = hit.material->albedo();
		if (hit.material->scatteredRay(ray, hit, scattered))
			return albedo * calculateRadiance(scattered, scene, depth - 1);
		else return Vec3(0.0f, 0.0f, 0.0f);
	}
	else return Vec3(0.95f, 0.95f, 1.0f); // background
}

void raytracing(int width, int height, Camera* camera, Shape* scene, std::ofstream& out, int sample, float gamma)
{
	// iterate through each pixel and write color to an output stream.
	for (int y = height - 1; y != 0; --y)
	{
		for (int x = 0; x != width; ++x)
		{
			// Stratified Sampling
			Vec3 color(0.0, 0.0, 0.0);
			for (int xi = 0; xi < sample; ++xi)
			{
				for (int yi = 0; yi < sample; ++yi)
				{
					float sx = float(x + ((xi + random_number())) / sample) / float(width);
					float sy = float(y + ((yi + random_number())) / sample) / float(height);
					Ray ray = camera->generateRay(sx, sy);
					color = color + calculateRadiance(ray, scene, 15);
				}
			}
			Vec3 setPixel = color;
			setPixel = color / float(sample * sample);
			int r = int(255 * (pow(setPixel[0], 1 / gamma)));
			int g = int(255 * (pow(setPixel[1], 1 / gamma)));
			int b = int(255 * (pow(setPixel[2], 1 / gamma)));
			out << r << " " << g << " " << b << "\n";
		}
	}
}

int main()
{
	Camera* camera = new Camera(6.0f, 3.0f);

	std::ofstream out("doc/cpp_test.ppm");
	out << "P3\n" << W << " " << H << "\n255\n";
	
	auto start = std::chrono::high_resolution_clock::now();
	Shape* objects[13];
	objects[0] = new Sphere(Vec3(0.0f, 0.0f, -1.0f), 0.5f, new Diffuse(Vec3(0.2f, 0.6f, 0.8f))); // center diffuse sphere
	objects[1] = new Sphere(Vec3(0.0f, 0.0f, 1.5f), 0.5f, new Diffuse(Vec3(1.0f, 0.0f, 1.0f))); // behind camera diffuse sphere
	objects[2] = new Sphere(Vec3(-0.20f, -0.45f, -0.65f), 0.05f, new Diffuse(Vec3(1.0f, 0.45f, 0.5f))); // pink diffuse sphere infront of center sphere
	objects[3] = new Sphere(Vec3(0.78f, -0.15f, -1.0f), 0.3f, new PolishedMetal(Vec3(1.0f, 1.0f, 1.0f), 0.33f)); // polished metal sphere right from center sphere
	objects[4] = new Sphere(Vec3(-0.78f, -0.15f, -1.0f), 0.3f, new Diffuse(Vec3(1.0f, 0.0f, 0.0f))); // red diffuse sphere
	objects[5] = new Sphere(Vec3(0.75f, -0.23f, -0.48f), 0.1f, new Mirror(Vec3(1.0f, 1.0f, 1.0f))); // mirror sphere down right
	objects[6] = new Sphere(Vec3(-0.75f, -0.23f, -0.48f), 0.1f, new Mirror(Vec3(1.0f, 1.0f, 1.0f))); // mirror sphere down left
	objects[7] = new Sphere(Vec3(0.29f, 0.2f, -0.39f), 0.05f, new Diffuse(Vec3(0.2f, 0.8f, 0.2f))); // green sphere up right
	objects[8] = new Sphere(Vec3(-0.29f, 0.2f, -0.39f), 0.05f, new PolishedMetal(Vec3(1.0f, 1.0f, 1.0f), 1.0f)); // polished metal sphere up left
	objects[9] = new Sphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f, new Diffuse(Vec3(0.85f, 0.85f, 0.85f))); // plane sphere
	objects[10] = new Sphere(Vec3(-0.43f, -0.40f, -0.85f), 0.05f, new Mirror(Vec3(1.0f, 0.0f, 1.0f))); // tiny purple mirror sphere 
	objects[11] = new Sphere(Vec3(0.40f, -0.40f, -0.75f), 0.09f, new Mirror(Vec3(1.0f, 1.0f, 0.0f))); // yellow mirror sphere
	objects[12] = new Sphere(Vec3(-0.15f, 0.21f, -0.56f), 0.06f, new Diffuse(Vec3(0.2f, 0.8f, 0.6f))); // aqua sphere on blue sphere
	Shape* scene = new Group(objects, 13);

	raytracing(W, H, camera, scene, out, SAMPLES, GAMMA);
	auto end = std::chrono::high_resolution_clock::now();
	std::cerr << "\nRendering took: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds\n";
}