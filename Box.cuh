#pragma once
#include "Figure.cuh"

class Box : virtual public Figure
{
private:
	float3 dimensions;

public:
	__device__ Box()
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimensions(make_float3(1.0f, 2.0f, 3.0f));
	}

	__device__ Box(float3 Dimensions)
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimensions(Dimensions);
	}

	__device__ Box(float3 Dimensions, float3 Position)
	{
		this->setColor(make_float3(128.0f, 128.0f, 128.0f));
		this->setPosition(Position);
		this->setDimensions(Dimensions);
	}

	__device__ Box(float3 Dimensions, float3 Position, float3 Color)
	{
		this->setColor(Color);
		this->setPosition(Position);
		this->setDimensions(Dimensions);
	}


	__device__ float draw(float3 pointPosition) override
	{
		float3 d = fabs(pointPosition) - getDimensions();
		return length(make_float3(max(d.x, 0.0f), max(d.y, 0.0f), max(d.z, 0.0f))) + min(max(d.x, max(d.y, d.z)), 0.0);
	}

	__device__ float3 getDimensions()
	{
		return this->dimensions;
	}

	__device__ void setDimensions(float3 Dimensions)
	{
		this->dimensions = Dimensions;
	}
};