#include "device_launch_parameters.h"

#include "library/Scene.cuh"
#include "library/Controls.cuh"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "library/Constants.cuh"

float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer


void time(int x)
{
	if (glutGetWindow())
	{
		glutPostRedisplay();
		glutTimerFunc(10, time, 0);
	}
}

void display()
{
	cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);
	dim3 block(16, 16, 1);
	dim3 grid(_width / block.x, _height / block.y, 1);
	rendering <<< grid, block >>> (device, rotX, rotY, cameraPos);   //execute kernel
	cudaGLUnmapBufferObject(buffer);

	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	glVertexPointer(2, GL_FLOAT, 12, 0);
	glColorPointer(4, GL_UNSIGNED_BYTE, 12, (GLvoid*)8);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glDrawArrays(GL_POINTS, 0, _width * _height);
	glDisableClientState(GL_VERTEX_ARRAY);
	glutSwapBuffers();
}

int main(int argc, char** argv)
{
	cudaMalloc(&device, _width * _height * sizeof(float3));   //allocate memory on the GPU VRAM

	glutInit(&argc, argv);   //OpenGL initializing
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowPosition(0, 0); // Start point
	glutInitWindowSize(_width, _height); // Window size
	glutCreateWindow("CUDA OpenGL raymarching"); // Window name

	glMatrixMode(GL_PROJECTION);
	gluOrtho2D(0.0, _width, 0.0, _height);
	glutDisplayFunc(display);
	
	// glutMouseFunc(mouseButton);
	// glutMotionFunc(mouseMove);

	glutKeyboardFunc(keyPressed);

	time(0);
	glewInit();
	glGenBuffers(1, &buffer);
	glBindBuffer(GL_ARRAY_BUFFER, buffer);
	unsigned int size = _width * _height * sizeof(float3);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	cudaGLRegisterBufferObject(buffer);   //register the buffer object for access by CUDA
	glutMainLoop();   //event processing loop

	cudaFree(device); //free memory on the GPU VRAM
}