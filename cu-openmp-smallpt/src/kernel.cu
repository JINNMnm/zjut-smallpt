//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "imageio.hpp"
#include "sampling.hpp"
#include "specular.hpp"
#include "sphere.hpp"
#include <chrono>
#include <omp.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#pragma endregion

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define REFRACTIVE_INDEX_OUT 1.0
#define REFRACTIVE_INDEX_IN  1.5
#define EPSILON_SPHERE 1e-4

#pragma endregion

using Clock = std::chrono::high_resolution_clock;

__constant__ Sphere dev_spheres[10];

__device__ bool Intersect(const Ray& ray, size_t& id) {
	bool hit = false;
	for (int i = 0; i < 10; ++i) {
		if (dev_spheres[i].Intersect(ray)) {
			hit = true;
			id = i;
		}
	}
	return hit;
}

__device__ Vector3 Radiance(Ray ray, curandState* state) {
	Vector3 L;
	Vector3 F(1.0);

	while (true) {
		size_t id;
		if (!Intersect(ray, id)) return L;

		const Sphere& shape = dev_spheres[id];
		const Vector3 p = ray(ray.m_tmax);
		const Vector3 n = Normalize(p - shape.m_p);

		L += F * shape.m_e;
		F *= shape.m_f;

		if (4 < ray.m_depth) {
			const double continue_probability = shape.m_f.Max();
			if (curand_uniform_double(state) >= continue_probability) return L;
			F /= continue_probability;
		}

		switch (shape.m_reflection_t) {
		case Reflection_t::Specular: {
			const Vector3 d = IdealSpecularReflect(ray.m_d, n);
			ray = Ray(p, d, EPSILON_SPHERE, INFINITY, ray.m_depth + 1);
			break;
		}
		case Reflection_t::Refractive: {
			double pr;
			const Vector3 d = IdealSpecularTransmit(ray.m_d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, pr, state);
			F *= pr;
			ray = Ray(p, d, EPSILON_SPHERE, INFINITY, ray.m_depth + 1);
			break;
		}
		default: {
			const Vector3 w = (0.0 > n.Dot(ray.m_d)) ? n : -n;
			const Vector3 u = Normalize((abs(w.m_x) > 0.1 ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0)).Cross(w));
			const Vector3 v = w.Cross(u);
			const Vector3 sample_d = CosineWeightedSampleOnHemisphere(curand_uniform_double(state), curand_uniform_double(state));
			const Vector3 d = Normalize(sample_d.m_x * u + sample_d.m_y * v + sample_d.m_z * w);
			ray = Ray(p, d, EPSILON_SPHERE, INFINITY, ray.m_depth + 1);
			break;
		}
		}
	}
}

__global__ void render_kernel(Vector3* Ls, int w, int h, int nb_samples, Vector3 eye, Vector3 cx, Vector3 cy, Vector3 gaze) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= w || y >= h) return;

	int i = (h - 1 - y) * w + x;

	curandState state;
	curand_init(i, 0, 0, &state);

	Vector3 L;
	for (int sy = 0; sy < 2; ++sy) {
		for (int sx = 0; sx < 2; ++sx) {
			Vector3 subL;
			for (int s = 0; s < nb_samples; ++s) {
				double u1 = 2.0 * curand_uniform_double(&state);
				double u2 = 2.0 * curand_uniform_double(&state);
				double dx = u1 < 1.0 ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
				double dy = u2 < 1.0 ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
				Vector3 d = cx * (((sx + 0.5 + dx) * 0.5 + x) / w - 0.5) +
				            cy * (((sy + 0.5 + dy) * 0.5 + y) / h - 0.5) + gaze;
				Ray r(eye + d * 130.0, Normalize(d), EPSILON_SPHERE);
				subL += Radiance(r, &state) * (1.0 / nb_samples);
			}
			L += 0.25 * Clamp(subL);
		}
	}
	Ls[i] = L;
}

namespace smallpt {
	static void RenderCUDA(std::uint32_t nb_samples) {
		const std::uint32_t w = 1024u;
		const std::uint32_t h = 768u;

		const Vector3 eye  = { 50.0, 52.0, 295.6 };
		const Vector3 gaze = Normalize(Vector3(0.0, -0.042612, -1.0));
		const double fov   = 0.5135;
		const Vector3 cx   = { w * fov / h, 0.0, 0.0 };
		const Vector3 cy   = Normalize(cx.Cross(gaze)) * fov;

		Vector3* dev_Ls;
		cudaMalloc(&dev_Ls, w * h * sizeof(Vector3));
		cudaMemset(dev_Ls, 0, w * h * sizeof(Vector3));

		cudaMemcpyToSymbol(dev_spheres, g_spheres, sizeof(g_spheres));

		dim3 block(16, 16);
		dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
		render_kernel<<<grid, block>>>(dev_Ls, w, h, nb_samples, eye, cx, cy, gaze);

		std::unique_ptr<Vector3[]> Ls(new Vector3[w * h]);
		cudaMemcpy(Ls.get(), dev_Ls, w * h * sizeof(Vector3), cudaMemcpyDeviceToHost);

		WritePPM(w, h, Ls.get(), nb_samples);

		cudaFree(dev_Ls);
	}
}

int main(int argc, char* argv[]) {
	const std::uint32_t nb_samples = (argc == 2) ? atoi(argv[1]) / 4 : 1;
	auto start = Clock::now();

	int device_count;
	cudaGetDeviceCount(&device_count);

	smallpt::RenderCUDA(nb_samples);

	auto end = Clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	std::cout << "\n[Render completed] Samples: " << nb_samples * 4
	          << " | Time elapsed: " << duration_ms << " ms" << std::endl;
	return 0;
}
