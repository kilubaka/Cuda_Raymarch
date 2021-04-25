#pragma once
#include <helper_math.h> 

class Figure
{
protected:
    float3 position;
    __device__ virtual float draw(float3 Position);

public:
    __device__ float3 getPosition()
    {
        return this->position;
    }

    __device__ void setPosition(float3 Position)
    {
        this->position = Position;
    }
};
