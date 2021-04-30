#pragma once
#include "Figure.cuh"

class Prism : virtual public Figure
{
private:
	float2 dimensions;

public:
	__device__ Prism()
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimensions(make_float2(1.0f, 2.0f));
	}

	__device__ Prism(float2 Dimensions)
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setDimensions(Dimensions);
	}

	__device__ Prism(float2 Dimensions, float3 Position)
	{
		this->setPosition(Position);
		this->setDimensions(Dimensions);
	}


	__device__ float draw(float3 pointPosition) override
	{
		pointPosition -= getPosition();
		float3 q = fabs(pointPosition);
		return max(q.z - getDimensions().y, max(q.x * 0.866025 + pointPosition.y * 0.5, - pointPosition.y) - getDimensions().x * 0.5);
	}

	__device__ float2 getDimensions()
	{
		return this->dimensions;
	}

	__device__ void setDimensions(float2 Dimensions)
	{
		this->dimensions = Dimensions;
	}
};