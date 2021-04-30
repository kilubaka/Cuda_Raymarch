#include "Operation.cuh"

class OpSubtraction : virtual public Operation
{
public:
	__device__ float operate() override
	{
		return max(-1 * getPrimitive_1(), getPrimitive_2());
	}

	__device__ float operate(float p_1, float p_2) override
	{
		return max(-1 * p_1, p_2);
	}

	__device__ OpSubtraction()
	{
		this->setPrimitive_1(0.0f);
		this->setPrimitive_2(0.0f);
	}

	__device__ OpSubtraction(float p_1, float p_2)
	{
		this->setPrimitive_1(p_1);
		this->setPrimitive_2(p_2);
	}
};