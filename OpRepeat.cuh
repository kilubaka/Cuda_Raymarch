#pragma once
#include <helper_math.h>

class OpRepeat
{
private:
	__device__ int Floor(float x)
	{
		int i = int(x); 
		if (i > x) i--;
		return i;
	}

	__device__ float mod(float x, float y) 
	{
		return x - y * Floor(x / y);
	}


public:
	__device__ float3 operate(float3 p)
	{
		p.x = mod(p.x, 1.0f) - 0.5f;
		p.y = mod(p.y, 1.0f) - 0.5f;
		p.z = mod(p.z, 1.0f) - 0.5f;

		return p;
	}

	__device__ float3 operate(float3 p, float cell_size)
	{
		p.x = mod(p.x, cell_size) - cell_size / 2;
		p.y = mod(p.y, cell_size) - cell_size / 2;
		p.z = mod(p.z, cell_size) - cell_size / 2;

		return p;
	}

	__device__ float3 operate(float3 p, bool is_x, bool is_y, bool is_z, float cell_size)
	{
		p.x = is_x ? mod(p.x, cell_size) - cell_size / 2 : p.x;
		p.y = is_y ? mod(p.y, cell_size) - cell_size / 2 : p.y;
		p.z = is_z ? mod(p.z, cell_size) - cell_size / 2 : p.z;

		return p;
	}
};