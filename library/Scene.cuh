#pragma once

#include "DistanceField.cuh"
#include "Constants.cuh"

__device__ float AmbientOcclusion(float3 p, float3 n, float3 cameraPos)
{
	float ao = 0.0;
	float dist;
	for (int i = 1; i <= _AoIterations; i++)
	{
		dist = _AoStepSize * i;
		ao += max(0.0f, (dist - distancefield(p + n * dist, cameraPos)) / dist);
	}
	return (1.0 - ao * _AoIntensity);
}

__device__ float softShadow(float3 ro, float3 rd, float minDt, float maxDt, float k, float3 cameraPos)
{
	float result = 1.0f;

	for (float t = minDt; t < maxDt;)
	{
		float h = distancefield(ro + rd * t, cameraPos);
		if (h < 0.001f)
		{
			return 0.0f;
		}
		result = min(result, k * h / t);
		t += h;
	}
	return result;
}

__device__ float3 Shading(float3 p, float3 n, float3 cameraPos)
{
	// Directional light
	float3 result = (_LightColor * dot(-1 * _LightDirection, n) * 0.5 + 0.5) * _LightIntensity;

	// Shadows
	float shadow = softShadow(p, -1 * _LightDirection, _ShadowDistance.x, _ShadowDistance.y, _ShadowPenumbra, cameraPos) * 0.5 + 0.5;
	shadow = max(0.0, pow(shadow, _ShadowIntensity));

	result *= _MainColor * shadow * AmbientOcclusion(p, n, cameraPos);
	return result;
}

// get normal from surface to shade properly
__device__ float3 getNormal(float3 position, float3 cameraPos)
{
	const float2 offset = make_float2(0.001f, 0.0f);

	float3 xyy = { offset.x, offset.y, offset.y };
	float3 yxy = { offset.y, offset.x, offset.y };
	float3 yyx = { offset.x, offset.y, offset.x };

	float3 normal = {
		distancefield(position + xyy, cameraPos) - distancefield(position - xyy, cameraPos),
		distancefield(position + yxy, cameraPos) - distancefield(position - yxy, cameraPos),
		distancefield(position + yyx, cameraPos) - distancefield(position - yyx, cameraPos)
	};

	return normalize(normal);
}

__device__ float3 raymarch(float3 ro, float3 rd, float depth, int maxIteration, float maxDistance, float3 p, float3 cameraPos)
{
	float3 result = { 0, 0, 0 };
	float distanceTravelled = 0.0f;    // distance what ray travelled

	for (int i = 0; i < maxIteration; i++) {
		if (distanceTravelled > maxDistance || distanceTravelled >= depth) {
			// draw environment, for endless rays
			// paint as color of direction
			result = rd;
			break;
		}

		// position
		p = ro + rd * distanceTravelled;

		//check for hit in distancefield, return distance
		float distance = distancefield(p, cameraPos);
		if (distance < _Accuracy) { // too close to think that ray hit

			result = p;
			break;
		}
		distanceTravelled += distance;
	}

	return result;
}

__global__ void rendering(float3* output, float rotX, float rotY, float3 cameraPos)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int i = (_height - y - 1) * _width + x;
	float2 resolution = { (float)_width, (float)_height };   //screen resolution
	float2 coordinates = { (float)x, (float)y };   //fragment coordinates


	float3 ro = make_float3(0.0f, 0.0f, -8.0f);   //ray origin

	float2 uv = (2 * coordinates - resolution) / resolution.y;
	float finalX = uv.x + rotX;
	float finalY = uv.y + rotY;
	float3 rd = normalize(make_float3(finalX, finalY, 1.0f));   //ray direction

	float3 c = rd;
	float3 hitPosition = raymarch(ro, rd, _maxDistance, _maxIteration, _maxDistance, ro, cameraPos);

	bool miss = rd.x == hitPosition.x && rd.y == hitPosition.y && rd.z == hitPosition.z;

	if (!miss)
	{
		// Shading
		float3 n = getNormal(hitPosition, cameraPos);
		float3 s = Shading(hitPosition, n, cameraPos);
		c = s;

		unsigned int mipLevel = 2;
		float invMipLevel = 0.5f;

		for (int i = 0; i < _ReflectionCount; i++) // Reflections
		{
			rd = normalize(reflect(rd, n));
			ro = hitPosition + rd * 0.01f;

			hitPosition = raymarch(ro, rd, _maxDistance * invMipLevel, _maxDistance * invMipLevel, _maxIteration / mipLevel, hitPosition, cameraPos);

			miss = rd.x == hitPosition.x && rd.y == hitPosition.y && rd.z == hitPosition.z;

			if (!miss)
			{
				n = getNormal(hitPosition, cameraPos);
				s = Shading(hitPosition, n, cameraPos);
				c += s * invMipLevel;
			}
			else break;

			mipLevel *= 2;
			invMipLevel *= 0.5f;
		}
	}


	unsigned char bytes[] = {
		(unsigned char)(c.x * 255),
		(unsigned char)(c.y * 255),
		(unsigned char)(c.z * 255),
		1,
	};
	float colour;
	memcpy(&colour, &bytes, sizeof(colour));   //convert from 4 bytes to single float
	output[i] = make_float3(x, y, colour);
}