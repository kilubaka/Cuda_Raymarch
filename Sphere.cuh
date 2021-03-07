#include "Figure.cuh"

class Sphere : virtual public Figure
{
private:
	float radius;

public:
	__device__ Sphere()
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setRadius(2.0f);
	}

	__device__ Sphere(float3 Position)
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(Position);
		this->setRadius(1.0f);
	}

	__device__ Sphere(float Radius)
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setRadius(Radius);
	}

	__device__ Sphere(float3 Position, float3 Color)
	{
		this->setColor(Color);
		this->setPosition(Position);
		this->setRadius(1.0f);
	}

	__device__ Sphere(float3 Position, float Radius)
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(Position);
		this->setRadius(Radius);
	}

	__device__ Sphere(float3 Color, float3 Position, float Radius)
	{
		this->setColor(Color);
		this->setPosition(Position);
		this->setRadius(Radius);
	}


	__device__ float draw(float3 pointPosition) override
	{
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