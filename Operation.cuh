#pragma once
#include <helper_math.h>

class Operation
{
protected:
    float primitive_1;
    float primitive_2;

public:
    __device__ virtual float operate();
    __device__ virtual float operate(float p_1, float p_2);

    __device__ float getPrimitive_1()
    {
        return this->primitive_1;
    }

    __device__ void setPrimitive_1(float Primitive_1)
    {
        this->primitive_1 = Primitive_1;
    }

    __device__ float getPrimitive_2()
    {
        return this->primitive_2;
    }

    __device__ void setPrimitive_2(float Primitive_2)
    {
        this->primitive_2 = Primitive_2;
    }
};