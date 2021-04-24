#pragma once
#include "Figure.cuh"

class Mandelbrot : virtual public Figure
{
private:
	int iterateCount, power;
	float bailout, scale;

public:
	__device__ Mandelbrot()
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setIterateCount(25);
		this->setPower(8);
		this->setBailout(4.0f);
		this->setScale(3.0f);
	}

	__device__ Mandelbrot(float scale)
	{
		this->setScale(scale);
	}

	__device__ Mandelbrot(int iterateCount, float3 Position)
	{
		this->setIterateCount(iterateCount);
		this->setPosition(Position);
	}

	__device__ Mandelbrot(int iterateCount, int power)
	{
		this->setIterateCount(iterateCount);
		this->setPower(power);
	}

	__device__ Mandelbrot(int iterateCount, int power, float3 Position)
	{
		this->setIterateCount(iterateCount);
		this->setPower(power);
		this->setPosition(Position);
	}

	__device__ float draw(float3 pointPosition) override
	{
		float3 p = pointPosition;
		p /= scale;
		float3 z = p;
		float dr = 4.0;
		float r = 0.0;
		float power = this->getPower();
		for (int i = 0; i < this->getIterateCount(); i++) {
			r = length(z);
			if (r > this->getBailout()) break;

			// convert to polar coordinates
			float theta = acos(z.z / r);
			float phi = atan2(z.y, z.x);
			dr = pow(r, power - 1.0) * power * dr + 1.0;

			// scale and rotate the point
			float zr = pow(r, power);
			theta = theta * power;
			phi = phi * power;

			// convert back to cartesian coordinates
			z = zr * make_float3(sin(theta) * cos(phi), sin(phi) * sin(theta), cos(theta));
			z += p;
		}
		return 0.5 * log(r) * r / dr * scale;
	}


	__device__ int getIterateCount()
	{
		return this->iterateCount;
	}

	__device__ void setIterateCount(int iterateCount)
	{
		this->iterateCount = iterateCount;
	}


	__device__ int getPower()
	{
		return this->power;
	}

	__device__ void setPower(int power)
	{
		this->power = power;
	}


	__device__ float getBailout()
	{
		return this->bailout;
	}

	__device__ void setBailout(float bailout)
	{
		this->bailout = bailout;
	}


	__device__ float getScale()
	{
		return this->scale;
	}

	__device__ void setScale(float scale)
	{
		this->scale = scale;
	}
};