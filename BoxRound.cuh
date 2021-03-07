#include "Box.cuh"

class BoxRound : public Box
{
private:
	float roundness;

public:
	__device__ BoxRound() : Box()
	{
		this->setRoundness(0.0f);
	}

	__device__ BoxRound(float Roundness) : Box()
	{
		this->setRoundness(Roundness);
	}

	__device__ BoxRound(float Roundness, float3 Dimensions) : Box(Dimensions)
	{
		this->setRoundness(Roundness);
	}

	__device__ BoxRound(float Roundness, float3 Dimensions, float3 Position) : Box(Dimensions, Position)
	{
		this->setRoundness(Roundness);
	}

	__device__ BoxRound(float Roundness, float3 Dimensions, float3 Position, float3 Color) : Box(Dimensions, Position, Color)
	{
		this->setRoundness(Roundness);
	}

	__device__ BoxRound(float3 Dimensions) : Box(Dimensions)
	{
		this->setRoundness(0.0f);
	}

	__device__ BoxRound(float3 Dimensions, float3 Position) : Box(Dimensions, Position)
	{
		this->setRoundness(0.0f);
	}

	__device__ BoxRound(float3 Dimensions, float3 Position, float3 Color) : Box(Dimensions, Position, Color)
	{
		this->setRoundness(0.0f);
	}


	__device__ float draw(float3 pointPosition) override
	{
		return Box::draw(pointPosition) - getRoundness();
	}

	__device__ float getRoundness()
	{
		return this->roundness;
	}

	__device__ void setRoundness(float Roundness)
	{
		this->roundness = Roundness;
	}
};