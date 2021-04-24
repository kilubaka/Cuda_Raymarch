#pragma once
#include "Figure.cuh"

class SierpinskiTriangle : virtual public Figure
{
private:
	int iterateCount;
	float offset;

public:
	__device__ SierpinskiTriangle()
	{
		this->setPosition(make_float3(0.0f, 0.0f, 0.0f));
		this->setIterateCount(25);
		this->setOffset(3.0f);
	}

	__device__ SierpinskiTriangle(int iterateCount)
	{
		this->setIterateCount(iterateCount);
	}

	__device__ SierpinskiTriangle(int iterateCount, float3 Position)
	{
		this->setIterateCount(iterateCount);
		this->setPosition(Position);
	}

	__device__ SierpinskiTriangle(int iterateCount, float offset)
	{
		this->setIterateCount(iterateCount);
		this->setOffset(offset);
	}

	__device__ SierpinskiTriangle(int iterateCount, float offset, float3 Position)
	{
		this->setIterateCount(iterateCount);
		this->setOffset(offset);
		this->setPosition(Position);
	}

	__device__ float draw(float3 pointPosition) override
	{
		float3 p = pointPosition;
		int n = 0;
		while (n < this->getIterateCount()) {
			if (p.x + p.y < 0) {
				float x = p.x;
				p.x = -p.y;
				p.y = -x;
			}
			if (p.x + p.z < 0) {
				float x = p.x;
				p.x = -p.z;
				p.z = -x;
			}
			if (p.y + p.z < 0) {
				float z = p.z;
				p.z = -p.y;
				p.y = -z;
			}
			p = p * 2.0f - this->getOffset();
			n++;
		}
		return (length(p)) * pow(2.0f, -float(n));
	}


	__device__ int getIterateCount()
	{
		return this->iterateCount;
	}

	__device__ void setIterateCount(int iterateCount)
	{
		this->iterateCount = iterateCount;
	}


	__device__ float getOffset()
	{
		return this->offset;
	}

	__device__ void setOffset(float offset)
	{
		this->offset = offset;
	}
};