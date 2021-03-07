#include "Box.cuh"

class BoxBound : public Box
{
private:
	float boundness;

public:
	__device__ BoxBound() : Box()
	{
		this->setBoundness(0.0f);
	}

	__device__ BoxBound(float Roundness) : Box()
	{
		this->setBoundness(Roundness);
	}

	__device__ BoxBound(float Roundness, float3 Dimensions) : Box(Dimensions)
	{
		this->setBoundness(Roundness);
	}

	__device__ BoxBound(float Roundness, float3 Dimensions, float3 Position) : Box(Dimensions, Position)
	{
		this->setBoundness(Roundness);
	}

	__device__ BoxBound(float Roundness, float3 Dimensions, float3 Position, float3 Color) : Box(Dimensions, Position, Color)
	{
		this->setBoundness(Roundness);
	}

	__device__ BoxBound(float3 Dimensions) : Box(Dimensions)
	{
		this->setBoundness(0.0f);
	}

	__device__ BoxBound(float3 Dimensions, float3 Position) : Box(Dimensions, Position)
	{
		this->setBoundness(0.0f);
	}

	__device__ BoxBound(float3 Dimensions, float3 Position, float3 Color) : Box(Dimensions, Position, Color)
	{
		this->setBoundness(0.0f);
	}


	__device__ float draw(float3 pointPosition) override
	{
		float3 p = pointPosition;
		float e = getBoundness();

		p = fabs(p) - getDimensions();
		float3 q = fabs(p + e) - e;
		return min(
			min(
			length(make_float3(max(p.x, 0.0f), max(q.y, 0.0f), max(q.z, 0.0f))) + min(max(p.x, max(q.y, q.z)), 0.0),
			length(make_float3(max(q.x, 0.0f), max(p.y, 0.0f), max(q.z, 0.0f))) + min(max(q.x, max(p.y, q.z)), 0.0)
			),
			length(make_float3(max(q.x, 0.0f), max(q.y, 0.0f), max(p.z, 0.0f))) + min(max(q.x, max(q.y, p.z)), 0.0)
		);
	}

	__device__ float getBoundness()
	{
		return this->boundness;
	}

	__device__ void setBoundness(float Boundness)
	{
		this->boundness = Boundness;
	}
};