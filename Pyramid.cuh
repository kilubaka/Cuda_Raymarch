#pragma once
#include "Figure.cuh"

class Pyramid : virtual public Figure
{
private:
	float dimension;

public:
	__device__ Pyramid()
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimension(1.0f);
	}

	__device__ Pyramid(float Dimension)
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimension(Dimension);
	}

	__device__ Pyramid(float Dimension, float3 Position)
	{
		this->setPosition(Position);
		this->setDimension(Dimension);
	}


	__device__ float draw(float3 pointPosition) override
	{
		float3 p = pointPosition - getPosition();
		float h = getDimension();

		float m2 = h * h + 0.25;

		p.x = fabs(p.x);
		p.z = fabs(p.z);

		if (p.z > p.x) {
			float temp = p.z;
			p.z = p.x;
			p.x = temp;
		}
		p.x -= 0.5f;
		p.z -= 0.5f;

		float3 q = { p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y};

		float s = max(-q.x, 0.0);
		float t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0f, 1.0f);

		float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
		float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);

		float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);

		int sign = 1;
		if (max(q.z, -p.y) < 0) sign = -1;

		return sqrt((d2 + q.z * q.z) / m2) * sign;
	}

	__device__ float getDimension()
	{
		return this->dimension;
	}

	__device__ void setDimension(float Dimension)
	{
		this->dimension = Dimension;
	}
};