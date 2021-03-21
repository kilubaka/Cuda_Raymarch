#pragma once
#include "Operation.cuh"

class SmoothOperation : public Operation
{
protected:
	float smooth;

public:
    __device__ virtual float operate(float p_1, float p_2, float s);

    __device__ float getSmooth()
    {
        return this->smooth;
    }

    __device__ void setSmooth(float Smooth)
    {
        this->smooth = Smooth;
    }
};