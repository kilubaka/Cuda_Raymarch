#include "device_launch_parameters.h"
#include <helper_math.h> 
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "Figures.cuh"
#include "Operations.cuh"

#define width 1024   //screen width
#define height 1024   //screen height

float t = 0.0f;   //timer
float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer


// Light and shading params
__constant__ float
	_LightIntensity = 1.0f, // 0.5 - 2.0
	_ShadowPenumbra = 60.0f, // 0 - 128
	_ShadowIntensity = 1.0f; // 0 - 4

__constant__ float2
	_ShadowDistance = { 0.1f, 64.0f };

__constant__ float3
	_MainColor = { 0.0f, 0.8f, 0.8f },	// 0.0 - 1.0 (rgb)
	_LightColor = { 0.8f, 0.8f, 0.8f }, // 0.0 - 1.0 (rgb)
	_LightDirection = { 1.0f, 0.0f, 1.0f }; // 0.0 - 1.0 (xyz)


// Ambient Occlusion params
__constant__ float
	_AoStepSize = 0.2f,	 // 0.01 - 5
	_AoIntensity = 0.2f; // 0 - 1

__constant__ int
	_AoIterations = 3; // 1 - 5


// Reflection params
__constant__ float
	_Accuracy = 0.001f, // 0.(0)1 - 0.1
	_maxDistance = 200.0f; // 0 - 99999

__constant__ int
	_ReflectionCount = 2, // 1 - 5
	_maxIteration = 256; // 1 - 99999



__device__ float necklase(float3 p, int N, float degree, Sphere sphere) {
	float result = sphere.draw(p);

	for (int i = 0; i < N; i++){
		float rad = 0.0174532925f * degree * i;

		Sphere temp = Sphere(make_float3(
			cos(rad) * sphere.getPosition().x - sin(rad) * sphere.getPosition().y,
			sin(rad) * sphere.getPosition().x + cos(rad) * sphere.getPosition().y,
			sphere.getPosition().z),
			sphere.getRadius());

		result = SmOpUnion().operate(result, temp.draw(p), 0.2f);
	}

	return result;
}

__device__ float distancefield(float3 p, float t, float3 cameraPos)
{
	p = make_float3(p.x, cos(cameraPos.x) * p.y + sin(cameraPos.x) * p.z, -sin(cameraPos.x) * p.y + cos(cameraPos.x) * p.z); //add rotation to see better
	p = make_float3(cos(cameraPos.y) * p.x - sin(cameraPos.y) * p.z, p.y, sin(cameraPos.y) * p.x + cos(cameraPos.y) * p.z);

	Sphere sphere1 = Sphere(make_float3(0.0f, 2.0f, 0.0f), 1.0f);

	return necklase(p, 8, 45, sphere1);
	//return sphere1.draw(p);
}


__device__ float AmbientOcclusion(float3 p, float3 n, float time, float3 cameraPos)
{
	float ao = 0.0;
	float dist;
	for (int i = 1; i <= _AoIterations; i++)
	{
		dist = _AoStepSize * i;
		ao += max(0.0f, (dist - distancefield(p + n * dist, time, cameraPos)) / dist);
	}
	return (1.0 - ao * _AoIntensity);
}

__device__ float softShadow(float3 ro, float3 rd, float minDt, float maxDt, float k, float time, float3 cameraPos)
{
	float result = 1.0f;

	for (float t = minDt; t < maxDt;)
	{
		float h = distancefield(ro + rd * t, time, cameraPos);
		if (h < 0.001f)
		{
			return 0.0f;
		}
		result = min(result, k * h / t);
		t += h;
	}
	return result;
}

__device__ float3 Shading(float3 p, float3 n, float time, float3 cameraPos)
{
	// Directional light
	float3 result = (_LightColor * dot(-1 * _LightDirection, n) * 0.5 + 0.5) * _LightIntensity;

	// Shadows
	float shadow = softShadow(p, -1 * _LightDirection, _ShadowDistance.x, _ShadowDistance.y, _ShadowPenumbra, time, cameraPos) * 0.5 + 0.5;
	shadow = max(0.0, pow(shadow, _ShadowIntensity));

	result *= _MainColor * shadow * AmbientOcclusion(p, n, time, cameraPos);
	return result;
}

// get normal from surface to shade properly
__device__ float3 getNormal(float3 position, float t, float3 cameraPos)
{
	const float2 offset = make_float2(0.001f, 0.0f);

	float3 xyy = { offset.x, offset.y, offset.y };
	float3 yxy = { offset.y, offset.x, offset.y };
	float3 yyx = { offset.x, offset.y, offset.x };

	float3 normal = {
		distancefield(position + xyy, t, cameraPos) - distancefield(position - xyy, t, cameraPos),
		distancefield(position + yxy, t, cameraPos) - distancefield(position - yxy, t, cameraPos),
		distancefield(position + yyx, t, cameraPos) - distancefield(position - yyx, t, cameraPos)
	};

	return normalize(normal);
}

__device__ float3 raymarch(float3 ro, float3 rd, float depth, int maxIteration, float maxDistance, float3 p, float t, float3 cameraPos)
{
	float3 result = { 0, 0, 0 };
	float distanceTravelled = 0.0f;    // distance what ray travelled

	for (int i = 0; i < maxIteration; i++) {
		if (distanceTravelled > maxDistance || distanceTravelled >= depth) {
			// draw environment, for endless rays
			// paint as color of direction
			result = rd;
			break;
		}

		// position
		p = ro + rd * distanceTravelled;

		//check for hit in distancefield, return distance
		float distance = distancefield(p, t, cameraPos);
		if (distance < _Accuracy) { // too close to think that ray hit

			result = p;
			break;
		}
		distanceTravelled += distance;
	}

	return result;
}

__global__ void rendering(float3* output, float t, float rotX, float rotY, float3 cameraPos)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;
	float2 resolution = { (float)width, (float)height };   //screen resolution
	float2 coordinates = { (float)x, (float)y };   //fragment coordinates


	float3 ro = make_float3(0.0f, 0.0f, -8.0f);   //ray origin

	float2 uv = (2 * coordinates - resolution) / resolution.y;
	float finalX = uv.x + rotX;
	float finalY = uv.y + rotY;
	float3 rd = normalize(make_float3(finalX, finalY, 1.0f));   //ray direction
	
	float3 c = rd;
	float3 hitPosition = raymarch(ro, rd, _maxDistance, _maxIteration, _maxDistance, ro, t, cameraPos);

	bool miss = rd.x == hitPosition.x && rd.y == hitPosition.y && rd.z == hitPosition.z;

	if (!miss)
	{
		// Shading
		float3 n = getNormal(hitPosition, t, cameraPos);
		float3 s = Shading(hitPosition, n, t, cameraPos);
		c = s;

		unsigned int mipLevel = 2;
		float invMipLevel = 0.5f;

		for (int i = 0; i < _ReflectionCount; i++) // Reflections
		{
			rd = normalize(reflect(rd, n));
			ro = hitPosition + rd * 0.01f;

			hitPosition = raymarch(ro, rd, _maxDistance * invMipLevel, _maxDistance * invMipLevel, _maxIteration / mipLevel, hitPosition, t, cameraPos);

			miss = rd.x == hitPosition.x && rd.y == hitPosition.y && rd.z == hitPosition.z;

			if (!miss)
			{
				n = getNormal(hitPosition, t, cameraPos);
				s = Shading(hitPosition, n, t, cameraPos);
				c += s * invMipLevel;
			}
			else break;

			mipLevel *= 2;
			invMipLevel *= 0.5f;
		}
	}

	
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


// Actual vector representing the camera's direction
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
	}
}


// Moving
float3 cameraPos = { 0.0f, 0.0f, 0.0f };

void keyPressed(unsigned char key, int x, int y) {
	switch (key)
	{
	case 'w': cameraPos.x += 0.1f; break;
	case 's': cameraPos.x -= 0.1f; break;
	case 'd': cameraPos.y -= 0.1f; break;
	case 'a': cameraPos.y += 0.1f; break;
	case 'r': cameraPos.z += 0.1f; break;
	case 'f': cameraPos.z -= 0.1f; break;
	default:
		break;
	}
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

void display()
{
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);
	dim3 block(16, 16, 1);
	dim3 grid(width / block.x, height / block.y, 1);
	rendering <<< grid, block >>> (device, t, rotX, rotY, cameraPos);   //execute kernel
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

	glutKeyboardFunc(keyPressed);

	time(0);
	glewInit();
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	unsigned int size = width * height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
	glutMainLoop();   //event processing loop

	cudaFree(device); //free memory on the GPU VRAM
}