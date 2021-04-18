#include "Figure.cuh"

class Sphere : virtual public Figure
{
private:
	float radius;

public:
	__device__ Sphere()
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setRadius(2.0f);
	}

	__device__ Sphere(float3 Position)
	{
		this->setPosition(Position);
		this->setRadius(1.0f);
	}

	__device__ Sphere(float Radius)
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setRadius(Radius);
	}

	__device__ Sphere(float3 Position, float Radius)
	{
		this->setPosition(Position);
		this->setRadius(Radius);
	}


	__device__ float draw(float3 pointPosition) override
	{
		pointPosition -= getPosition();
		return length(pointPosition - this->getPosition()) - this->getRadius();
	}

	__device__ float getRadius()
	{
		return this->radius;
	}

	__device__ void setRadius(float Radius)
	{
		this->radius = Radius < 0.0f ? 0.0f : Radius;
	}
};