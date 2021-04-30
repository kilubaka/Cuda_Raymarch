#pragma once

#include "device_launch_parameters.h"
#include "../inc/helper_math.h" 
#include "../GL/glew/include/GL/glew.h"
#include "../GL/freeglut/include/GL/freeglut.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Figures.cuh"
#include "Operations.cuh"

__device__ float distancefield(float3 p, float3 cameraPos)
{
	p = make_float3(p.x, cos(cameraPos.x) * p.y + sin(cameraPos.x) * p.z, -sin(cameraPos.x) * p.y + cos(cameraPos.x) * p.z); //add rotation to see better
	p = make_float3(cos(cameraPos.y) * p.x - sin(cameraPos.y) * p.z, p.y, sin(cameraPos.y) * p.x + cos(cameraPos.y) * p.z);

	Mandelbrot a = Mandelbrot();
	Box g = Box();

	return g.draw(p);
}