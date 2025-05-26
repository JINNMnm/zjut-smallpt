#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <cstdio>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	inline void WritePPM(std::uint32_t w, 
						 std::uint32_t h, 
						 const Vector3* Ls, 
						 std::uint32_t epoch = 0,
						 const char* prefix = "cpp-image.ppm") noexcept {
		char filename[256];
		std::snprintf(filename, sizeof(filename), "%s-epoch%u.ppm", prefix, epoch);

		FILE* fp = fopen(filename, "w");
		
		std::fprintf(fp, "P3\n%u %u\n%u\n", w, h, 255u);
		for (std::size_t i = 0; i < w * h; ++i) {
			std::fprintf(fp, "%u %u %u ", 
						 ToByte(Ls[i].m_x), 
						 ToByte(Ls[i].m_y), 
						 ToByte(Ls[i].m_z));
		}
		
		std::fclose(fp);
	}
}
