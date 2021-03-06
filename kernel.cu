#include "device_launch_parameters.h"
#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define width 1024   //screen width
#define height 1024   //screen height

float3* device;   //pointer to memory on the device (GPU VRAM)
GLuint buffer;   //buffer


void display(void)
{
	cudaThreadSynchronize();
	cudaGLMapBufferObject((void**)&device, buffer);   //maps the buffer object into the address space of CUDA
	glClear(GL_COLOR_BUFFER_BIT);

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