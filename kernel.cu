#include "device_launch_parameters.h"
#include <helper_math.h> 
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <iostream>

#include "Sphere.cuh"

#define width 1024   //screen width
#define height 1024   //screen height

float t = 0.0f;   //timer
float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer



__device__ float distancefield(float3 p)
{
	Sphere sp1;

	return sp1.draw(p);
}

__device__ float3 raymarch(float3 ro, float3 rd)   //raymarching
{
	const int maxIteration = 128;

	for (int i = 0; i < maxIteration; i++)
	{
		float d = distancefield(ro);
		if (d < 0.01) return make_float3(1.0f, 1.0f, 1.0f); // if collide paint white
		ro += d * rd;
	}
	return rd;   //background color
}

__global__ void rendering(float3* output, float k, float rotX, float rotY)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;
	float2 resolution = make_float2((float)width, (float)height);   //screen resolution
	float2 coordinates = make_float2((float)x, (float)y);   //fragment coordinates

	float2 uv = (2.0 * coordinates - resolution) / resolution.y;
	float3 ro = make_float3(0.0f, 0.0f, -8.0f);   //ray origin

	float finalX = uv.x + rotX;
	float finalY = uv.y + rotY;

	float3 rd = normalize(make_float3(finalX, finalY, 2.0f));   //ray direction
	float3 c = raymarch(ro, rd);
	
	unsigned char bytes[] = { 
		(unsigned char)(c.x * 255), 
		(unsigned char)(c.y * 255), 
		(unsigned char)(c.z * 255), 
		1,
	};
	float colour;
	memcpy(&colour, &bytes, sizeof(colour));   //convert from 4 bytes to single float
	output[i] = make_float3(x, y, colour);
}

void time(int x)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(10, time, 0);
		t += 0.01f;
	}
}


// actual vector representing the camera's direction
float rotX = 0.0f, rotY = 0.0f;
float mouseSensivityX = 0.01f, mouseSensivityY = 0.003f;

int mouseState = 0;

void mouseButton(int button, int state, int x, int y) {

	// only start motion if the left button is pressed
	if (button == GLUT_LEFT_BUTTON) {

		// when the button is pressed
		mouseState = state == GLUT_DOWN ? 1 : 0;
	}

	if (button == GLUT_RIGHT_BUTTON) {
		mouseState = state == GLUT_DOWN ? -1 : 0;
	}

}

void mouseMove(int x, int y) {
	glutWarpPointer(width / 2, height / 2);

	// this will only be true when the right button is down
	if (mouseState == -1) {
		// update camera's direction
		rotX += (x - width / 2) * mouseSensivityX;
		rotY += -(y - height / 2) * mouseSensivityY;

		if (fabs(rotX) > 180) rotX = clamp(-rotX, -180.0f, 180.0f);
		if (fabs(rotY) > 180) rotY = clamp(-rotY, -180.0f, 180.0f);

		std::cout << "rotX: " << rotX << "\trotY: " << rotY << std::endl;
	}
}


void display(void)
{
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	rendering <<< grid, block >>> (device, t, rotX, rotY);   //execute kernel
	cudaThreadSynchronize();
	cudaGLUnmapBufferObject(buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, width * height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	cudaMalloc(&device, width * height * sizeof(float3));   //allocate memory on the GPU VRAM
	glutInit(&argc, argv);   //OpenGL initializing
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(0, 0); // Start point
	glutInitWindowSize(width, height); // Window size
	glutCreateWindow("CUDA OpenGL raymarching"); // Window name

	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, width, 0.0, height);
	glutDisplayFunc(display);
	glutMouseFunc(mouseButton);
	glutMotionFunc(mouseMove);

	time(0);
	glewInit();
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	unsigned int size = width * height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
	glutMainLoop();   //event processing loop
	cudaFree(device);
}