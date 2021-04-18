#pragma once
#include "Scene.cuh"


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