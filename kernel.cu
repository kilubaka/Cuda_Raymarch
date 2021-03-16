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


__device__ float distancefield(float3 p, float t)
{
	p = make_float3(p.x, cos(t) * p.y + sin(t) * p.z, -sin(t) * p.y + cos(t) * p.z); //add rotation to see better
	p = make_float3(cos(t) * p.x - sin(t) * p.z, p.y, sin(t) * p.x + cos(t) * p.z);

	Box box1 = Box(make_float3(1.0f, 1.0f, 1.0f));
	Sphere sphere1 = Sphere(1.3f);


	return SmOpUnion().operate(sphere1.draw(p), box1.draw(p), 0.1f);
}

__device__ float softShadow(float3 ro, float3 rd, float minDt, float maxDt, float k, float time)
{
	float result = 1.0;

	for (float t = minDt; t < maxDt;)
	{
		float h = distancefield(ro + rd * t, time);
		if (h < 0.001)
		{
			return 0.0;
		}
		result = min(result, k * h / t);
		t += h;
	}
	return result;
}

__device__ float3 Shading(float3 p, float3 n, float time)
{
	const float3 
			_LightColor = make_float3(0.8f, 0.8f, 0.8f),
			_LightDirection = make_float3(1.0f, 0.0f, 1.0f);

	const float2
			_ShadowDistance = make_float2(0.1f, 10.0f);

	const float
			_LightIntensity = 1.0f,
			_ShadowPenumbra = 60.0f,
			_ShadowIntensity = 1.0f;


	// Directional light
	float3 result = (_LightColor * dot(-1 * _LightDirection, n) * 0.5 + 0.5) * _LightIntensity;

	// Shadows
	float shadow = softShadow(p, -1 * _LightDirection, _ShadowDistance.x, _ShadowDistance.y, _ShadowPenumbra, time) * 0.5 + 0.5;
	shadow = max(0.0, pow(shadow, _ShadowIntensity));
	result *= shadow;
	return result;
}

// get normal from surface to shade properly
__device__ float3 getNormal(float3 position, float t)
{
	const float2 offset = make_float2(0.001f, 0.0f);

	float3 xyy = make_float3(offset.x, offset.y, offset.y);
	float3 yxy = make_float3(offset.y, offset.x, offset.y);
	float3 yyx = make_float3(offset.x, offset.y, offset.x);

	float3 normal = make_float3(
		distancefield(position + xyy, t) - distancefield(position - xyy, t),
		distancefield(position + yxy, t) - distancefield(position - yxy, t),
		distancefield(position + yyx, t) - distancefield(position - yyx, t));

	return normalize(normal);
}

__device__ float3 raymarch(float3 ro, float3 rd, float t)
{
	const float3 _mainColor = make_float3(0.8f, 0.0f, 0.8f);
	const float _maxDistance = 200.0f;
	const int maxIteration = 256;

	float3 result = make_float3(1, 1, 1);
	float distanceTravelled = 0;    // distance what ray travelled

	for (int i = 0; i < maxIteration; i++) {
		if (distanceTravelled > 256.0f) {
			// draw environment, for endless rays
			// paint as color of direction
			result = rd;
			break;
		}

		// position
		float3 position = ro + rd * distanceTravelled;

		//check for hit in distancefield, return distance
		float distance = distancefield(position, t);
		if (distance < 0.01f) { // too close to think that ray hit

			float3 normal = getNormal(position, t);
			float3 light = Shading(position, normal, t); // product of 2 vectors

			result = _mainColor * light;
			break;
		}
		distanceTravelled += distance;
	}

	return result;
}

__global__ void rendering(float3* output, float t, float rotX, float rotY, float3 camPos)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (height - y - 1) * width + x;
	float2 resolution = make_float2((float)width, (float)height);   //screen resolution
	float2 coordinates = make_float2((float)x, (float)y);   //fragment coordinates

	float2 uv = (2 * coordinates - resolution) / resolution.y;
	float3 ro = make_float3(camPos.x, camPos.y, camPos.z);   //ray origin

	float finalX = uv.x + rotX;
	float finalY = uv.y + rotY;

	float3 rd = normalize(make_float3(finalX, finalY, 1.0f));   //ray direction
	float3 c = raymarch(ro, rd, t);
	
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
	}
}


// Moving
float3 cameraPos = make_float3(0.0f, 0.0f, -8.0f);

void keyPressed(unsigned char key, int x, int y) {
	if (key == 'w') cameraPos.z += 0.1f;
	if (key == 's') cameraPos.z -= 0.1f;
	if (key == 'd') cameraPos.x += 0.1f;
	if (key == 'a') cameraPos.x -= 0.1f;
	if (key == 'r') cameraPos.y += 0.1f;
	if (key == 'f') cameraPos.y -= 0.1f;
}


void display(void)
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
	cudaFree(device);
}