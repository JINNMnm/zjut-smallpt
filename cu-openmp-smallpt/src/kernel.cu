//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "imageio.hpp"
#include "sampling.cuh"
#include "specular.cuh"
#include "sphere.hpp"

#include "cuda_tools.hpp"
#include <cuda_runtime.h>
#include <omp.h>
#include <chrono>
#include <iostream>
using Clock = std::chrono::high_resolution_clock;

#pragma endregion

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define REFRACTIVE_INDEX_OUT 1.0
#define REFRACTIVE_INDEX_IN  1.5

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//__constant__ Sphere dev_spheres[9];

	const Sphere g_spheres[] = {
		Sphere(1e5,  Vector3(1e5 + 1, 40.8, 81.6),   Vector3(),   Vector3(0.15,0.85,0.25), Reflection_t::Diffuse),  //Left
		Sphere(1e5,  Vector3(-1e5 + 99, 40.8, 81.6), Vector3(),   Vector3(0.13,0.73,0.81), Reflection_t::Diffuse),  //Right
		Sphere(1e5,  Vector3(50, 40.8, 1e5),         Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),  //Back
		Sphere(1e5,  Vector3(50, 40.8, -1e5 + 170),  Vector3(),   Vector3(0.75),               Reflection_t::Diffuse),  //Front
		Sphere(1e5,  Vector3(50, 1e5, 81.6),         Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),  //Bottom
		Sphere(1e5,  Vector3(50, -1e5 + 81.6, 81.6), Vector3(),   Vector3(0.75),           Reflection_t::Diffuse),  //Top
		Sphere(16.5, Vector3(27, 16.5, 47),          Vector3(),   Vector3(0.999),          Reflection_t::Specular),  //Mirror
		Sphere(16.5, Vector3(73, 16.5, 78),          Vector3(),   Vector3(0.999),          Reflection_t::Diffuse),//Glass
		Sphere(8.5, Vector3(73, 53.5, 78),          Vector3(),   Vector3(0.999),          Reflection_t::Refractive),//Glass
		Sphere(600,  Vector3(50, 681.6 - .27, 81.6), Vector3(12), Vector3(),               Reflection_t::Diffuse)  //Light	
	};

	__device__ inline bool Intersect(const Sphere* dev_spheres, 
									 std::size_t nb_spheres, 
									 const Ray& ray, 
									 size_t& id) {
		
		bool hit = false;
		for (std::size_t i = 0u; i < nb_spheres; ++i) {
			if (dev_spheres[i].Intersect(ray)) {
				hit = true;
				id  = i;
			}
		}

		return hit;
	}

	__device__ static Vector3 Radiance(const Sphere* dev_spheres, 
									   std::size_t nb_spheres,
									   const Ray& ray, 
									   curandState* state) {
		
		Ray r = ray;
		Vector3 L;
		Vector3 F(1.0);

		while (true) {
			std::size_t id;
			if (!Intersect(dev_spheres, nb_spheres, r, id)) {
				return L;
			}

			const Sphere& shape = dev_spheres[id];
			const Vector3 p = r(r.m_tmax);
			const Vector3 n = Normalize(p - shape.m_p);

			L += F * shape.m_e;
			F *= shape.m_f;

			// Russian roulette
			if (4 < r.m_depth) {
				const double continue_probability = shape.m_f.Max();
				if (curand_uniform_double(state) >= continue_probability) {
					return L;
				}
				F /= continue_probability;
			}

			// Next path segment
			switch (shape.m_reflection_t) {
			
			case Reflection_t::Specular: {
				const Vector3 d = IdealSpecularReflect(r.m_d, n);
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
				break;
			}
			
			case Reflection_t::Refractive: {
				double pr;
				const Vector3 d = IdealSpecularTransmit(r.m_d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, pr, state);
				F *= pr;
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
				break;
			}
			
			default: {
				const Vector3 w = (0.0 > n.Dot(r.m_d)) ? n : -n;
				const Vector3 u = Normalize((abs(w.m_x) > 0.1 ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0)).Cross(w));
				const Vector3 v = w.Cross(u);

				const Vector3 sample_d = CosineWeightedSampleOnHemisphere(curand_uniform_double(state), curand_uniform_double(state));
				const Vector3 d = Normalize(sample_d.m_x * u + sample_d.m_y * v + sample_d.m_z * w);
				r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.m_depth + 1u);
			}
			}
		}
	}
	__global__ static void kernel(const Sphere* dev_spheres,
		std::size_t nb_spheres,
		std::uint32_t w,
		std::uint32_t h_full,
		std::uint32_t h_local,
		Vector3* Ls,
		std::uint32_t nb_samples,
		std::uint32_t y_offset) {
			const std::uint32_t x = threadIdx.x + blockIdx.x * blockDim.x;
			const std::uint32_t y_local = threadIdx.y + blockIdx.y * blockDim.y;

			if (x >= w || y_local >= h_local) return;

			const std::uint32_t y = y_offset + y_local;
			const std::uint32_t offset = x + y_local * w;

			// RNG
			curandState state;
			curand_init(offset, 0u, 0u, &state);

			const Vector3 eye = { 50.0, 52.0, 295.6 };
			const Vector3 gaze = Normalize(Vector3(0.0, -0.042612, -1.0));
			const double fov = 0.5135;
			const Vector3 cx = { w * fov / h_full, 0.0, 0.0 };
			const Vector3 cy = Normalize(cx.Cross(gaze)) * fov;

			std::size_t i = (h_local - 1u - y_local) * w + x;

			for (std::size_t sy = 0u; sy < 2u; ++sy) {
				for (std::size_t sx = 0u; sx < 2u; ++sx) {
					Vector3 L;

					for (std::size_t s = 0u; s < nb_samples; ++s) {
						const double u1 = 2.0 * curand_uniform_double(&state);
						const double u2 = 2.0 * curand_uniform_double(&state);
						const double dx = (u1 < 1.0) ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
						const double dy = (u2 < 1.0) ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
						const Vector3 d = cx * (((sx + 0.5 + dx) * 0.5 + x) / w - 0.5) +
									cy * (((sy + 0.5 + dy) * 0.5 + y) / h_full - 0.5) + gaze;

						L += Radiance(dev_spheres, nb_spheres,
							Ray(eye + d * 130, Normalize(d), EPSILON_SPHERE), &state)
						* (1.0 / nb_samples);
					}

				Ls[i] += 0.25 * Clamp(L);
			}
			}
		}

    static void Render_OpenMP(std::uint32_t nb_samples) noexcept {
        const std::uint32_t w = 1024u;
        const std::uint32_t h = 768u;
    
        Vector3* Ls_total = new Vector3[w * h];  // 最终图像
    
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
    
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            int device_id = tid % device_count;
            cudaSetDevice(device_id);
            #pragma omp critical
            {
                printf("Thread %d using GPU %d\n", tid, device_id);
                fflush(stdout);
            }
                
            const std::uint32_t h_local = h / nthreads;
            const std::uint32_t y_start = tid * h_local;
            const std::uint32_t nb_pixels_local = w * h_local;
    
            // 分配设备内存
            Sphere* dev_spheres;
            cudaMalloc(&dev_spheres, sizeof(g_spheres));
            cudaMemcpy(dev_spheres, g_spheres, sizeof(g_spheres), cudaMemcpyHostToDevice);
    
            Vector3* dev_Ls;
            cudaMalloc(&dev_Ls, nb_pixels_local * sizeof(Vector3));
            cudaMemset(dev_Ls, 0, nb_pixels_local * sizeof(Vector3));
    
            dim3 nblocks((w + 15) / 16, (h_local + 15) / 16);
            dim3 nthreads_block(16, 16);
    
            kernel<<<nblocks, nthreads_block>>>(
                dev_spheres,
                sizeof(g_spheres) / sizeof(g_spheres[0]),
                w, h, h_local,
                dev_Ls,
                nb_samples,
                y_start
            );
    
            // 拷贝结果回 Host
            Vector3* Ls_local = new Vector3[nb_pixels_local];
            cudaMemcpy(Ls_local, dev_Ls, nb_pixels_local * sizeof(Vector3), cudaMemcpyDeviceToHost);
    
            // 合并进全图像缓存
            for (std::uint32_t y = 0; y < h_local; ++y) {
                for (std::uint32_t x = 0; x < w; ++x) {
                    std::size_t global_idx = (y_start + y) * w + x;
                    std::size_t local_idx = y * w + x;
                    #pragma omp critical
                    Ls_total[global_idx] = Ls_local[local_idx];
                }
            }
    
            delete[] Ls_local;
            cudaFree(dev_Ls);
            cudaFree(dev_spheres);
        }
    
        WritePPM(w, h, Ls_total);
        delete[] Ls_total;
    }
}

int main(int argc, char* argv[]) {
	auto start = Clock::now();
	const std::uint32_t nb_samples = (argc == 2) ? atoi(argv[1]) / 4 : 1;

	smallpt::Render_OpenMP(nb_samples);

	auto end = Clock::now();
	auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

	std::cout << "\n[Render completed] Samples: " << nb_samples * 4
	          << " | Time elapsed: " << duration_ms << " ms" << std::endl;

	return 0;
}