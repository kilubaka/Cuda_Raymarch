#include "SmoothOperation.cuh"

class SmOpSubtraction : virtual public SmoothOperation
{
public:
	__device__ float operate() override
	{
		float h = clamp(0.5 - 0.5 * (getPrimitive_2() + getPrimitive_1()) / this->getSmooth(), 0.0f, 1.0f);
		return lerp(getPrimitive_2(), -getPrimitive_1(), h) + this->getSmooth() * h * (1.0 - h);
	}

	__device__ float operate(float p_1, float p_2) override
	{
		float h = clamp(0.5 - 0.5 * (p_2 + p_1) / this->getSmooth(), 0.0f, 1.0f);
		return lerp(p_2, -p_1, h) + this->getSmooth() * h * (1.0 - h);
	}

	__device__ float operate(float p_1, float p_2, float s) override
	{
		float h = clamp(0.5 - 0.5 * (p_2 + p_1) / s, 0.0f, 1.0f);
		return lerp(p_2, -p_1, h) + s * h * (1.0 - h);
	}

	__device__ SmOpSubtraction()
	{
		this->setPrimitive_1(0.0f);
		this->setPrimitive_2(0.0f);
		this->setSmooth(0.5f);
	}

	__device__ SmOpSubtraction(float p_1, float p_2)
	{
		this->setPrimitive_1(p_1);
		this->setPrimitive_2(p_2);
		this->setSmooth(0.5f);
	}

	__device__ SmOpSubtraction(float p_1, float p_2, float s)
	{
		this->setPrimitive_1(p_1);
		this->setPrimitive_2(p_2);
		this->setSmooth(s);
	}
};