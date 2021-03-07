#include <helper_math.h> 

class Figure
{
protected:
    float3 color;
    float3 position;

private:
    __device__ float virtual draw(float3 Position);

public:
    __device__ float3 getColor()
    {
        return this->color;
    }

    __device__ void setColor(float3 Color)
    {
        this->color = Color;
    }

    __device__ float3 getPosition()
    {
        return this->position;
    }

    __device__ void setPosition(float3 Position)
    {
        this->position = Position;
    }
};

