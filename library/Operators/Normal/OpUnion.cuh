#include "Operation.cuh"

class OpUnion : virtual public Operation
{
public:
    __device__ float operate() override
	{
		return min(getPrimitive_1(), getPrimitive_2());
	}

    __device__ float operate(float p_1, float p_2) override
	{
		return min(p_1, p_2);
	}

	__device__ OpUnion()
	{
		this->setPrimitive_1(0.0f);
		this->setPrimitive_2(0.0f);
	}

	__device__ OpUnion(float p_1, float p_2)
	{
		this->setPrimitive_1(p_1);
		this->setPrimitive_2(p_2);
	}
};