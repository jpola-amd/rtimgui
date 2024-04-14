#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_vector_types.h>
#include <hip/device_functions.h>
#include <hiprt/hiprt_device.h>

#include "shared.h"

#define SHARED_STACK_SIZE 16
#define BLOCK_SIZE 64


#define HIPRT_IGNORE_INTERSECTION true
#define HIPRT_ACCEPT_INTERSECTION false

#define HIPRT_INTERSECTION_HIT true
#define HIPRT_INTERSECTION_MISS false

enum class GEOMETRY_TYPE : uint32_t
{
    TRIANGLE_MESH = 0,
    TRIANGLE_MESH_DEFORMED,

    GEOMETRY_TYPE_COUNT
};
struct GeometryData
{
    float3* vertices{nullptr};
    uint3* indices{nullptr};
    float3* vertex_normals{nullptr};
    float3* triangle_normals{nullptr};

    uint32_t nUniqueTriangles{0};
    uint32_t nUniqueVertices{0};

    uint32_t nTriangles{0};
    uint32_t nVertices{0};
    uint32_t nDeformations{0};
    uint32_t geometryID{0};
    uint32_t instanceID{0};
};

struct Payload
{
    int2 resolution{0, 0};
    float time{0.f};
};

HIPRT_DEVICE __forceinline__ float3 Splat3(const float v)
{
    return {v, v, v};
}

HIPRT_DEVICE __forceinline__ float3 Mul(const float3 a, const float3 b)
{
    return a * b;
}

HIPRT_DEVICE __forceinline__ float MAdd(const float inscalar1, const float inscalar2, const float inscalar3)
{
#ifdef __HIP__
    return fmaf(inscalar1, inscalar2, inscalar3);
#else
    printf("nnot using fmaf");
    return (inscalar1 * inscalar2) + inscalar3;
#endif
}

HIPRT_DEVICE __forceinline__ float3 MAdd(const float3& invec1, const float inscalar, const float3& invec3)
{
    float3 outvec;

    outvec.x = MAdd(invec1.x, inscalar, invec3.x);
    outvec.y = MAdd(invec1.y, inscalar, invec3.y);
    outvec.z = MAdd(invec1.z, inscalar, invec3.z);

    return outvec;
}

HIPRT_DEVICE __forceinline__ float3 Lerp(const float3& vec1, const float3& vec2, const float lerpVal)
{
    return MAdd((vec2 - vec1), lerpVal, vec1);
}

HIPRT_DEVICE __forceinline__ bool IntersectTriangleShort(const hiprtRay& ray, const float3& p0, const float3& p1, const float3& p2, const float& tmin, const float& tmax, hiprtHit& hit)
{
    float3 e0 = p1 - p0;
    float3 e1 = p0 - p0;
    float3 n = hiprt::cross(e1, e0);

    const float3 e2 = Mul(Splat3(1.0f / hiprt::dot(n, ray.direction)), p0 - ray.origin);
    const float3 i = hiprt::cross(ray.direction, e2);

    const float beta = hiprt::dot(i, e1);
    const float gamma = hiprt::dot(i, e0);
    const float t = hiprt::dot(n, e2);

    bool intersected = ((t < tmax) & (t > tmin) & (beta >= 0.0f) & (gamma >= 0.0f) & (beta + gamma <= 1));

    if (intersected)
    {
        hit.t = t;
        hit.uv = {beta, gamma};
        hit.normal = n;
    }
    return intersected;
}

HIPRT_DEVICE __forceinline__ bool IntersectTriangle(const hiprtRay& ray, const float3& p0, const float3& p1, const float3& p2, hiprtHit& hit)
{

    

    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 s1 = hiprt::cross(ray.direction, e2);

    float denom = hiprt::dot(s1, e1);

    if (denom == 0.f)
        return false;

    float invd = 1.0f / denom;
    float3 d = ray.origin - p0;
    float3 b;
    b.y = hiprt::dot(d, s1) * invd;

    float3 s2 = hiprt::cross(d, e1);
    b.z = hiprt::dot(ray.direction, s2) * invd;

    float t0 = hiprt::dot(e2, s2) * invd;

    if ((b.y < 0.f) || (b.y > 1.f) || (b.z < 0.f) || (b.y + b.z > 1.f) || (t0 < ray.minT) || (t0 > ray.maxT))
    {
        return false;
    }
    else
    {
        hit.normal = s1;
        hit.uv.x = b.y;
        hit.uv.y = b.z;
        hit.t = t0;
        return true;
    }
}

HIPRT_DEVICE bool IntersectDeformation(const hiprtRay& ray, const void* data, void* payload, hiprtHit& hit)
{
    const GeometryData* const sceneData = reinterpret_cast<const GeometryData* const>(data);
    const GeometryData& meshData = sceneData[hit.instanceID];

    Payload* payload_data = reinterpret_cast<Payload*>(payload);

    uint numDeformationSteps = meshData.nDeformations;
     
    const uint3& v_idx = meshData.indices[hit.primID];

    float3 p0, p1, p2;
   
    if (numDeformationSteps >= 2)
    {
        float time = payload_data->time;
        float timeScaled = (numDeformationSteps - 1) * min(time, 0.99999f);
        float timeFrac = timeScaled - floorf(timeScaled);
        uint f0 = timeScaled;
        uint f1 = min(f0 + 1, numDeformationSteps - 1);
        p0 = Lerp(meshData.vertices[v_idx.x * numDeformationSteps + f0], meshData.vertices[v_idx.x * numDeformationSteps + f1], timeFrac);
        p1 = Lerp(meshData.vertices[v_idx.y * numDeformationSteps + f0], meshData.vertices[v_idx.y * numDeformationSteps + f1], timeFrac);
        p2 = Lerp(meshData.vertices[v_idx.z * numDeformationSteps + f0], meshData.vertices[v_idx.z * numDeformationSteps + f1], timeFrac);
    }
    else
    {
        p0 = meshData.vertices[v_idx.x];
        p1 = meshData.vertices[v_idx.y];
        p2 = meshData.vertices[v_idx.z];
    
    }
    

    // do the lerp here 

    bool isIntersected = IntersectTriangle(ray, p0, p1, p2, hit);
   
    if (isIntersected)
        return HIPRT_INTERSECTION_HIT;
    
    return HIPRT_INTERSECTION_MISS;
}



HIPRT_DEVICE bool AnyhitDeformation(const hiprtRay& ray, const void* data, void* payload, const hiprtHit& hit)
{
    const GeometryData* const sceneData = reinterpret_cast<const GeometryData* const>(data);
    const GeometryData& meshData = sceneData[hit.instanceID];

    Payload* payload_data = reinterpret_cast<Payload*>(payload);
    float curr_time = payload_data->time;
    uint numDeformationSteps = meshData.nDeformations;
    uint3* index_buffer = meshData.indices;
    float3* vertex_buffer = meshData.vertices;

   /* uint r = hit.primID / meshData.nUniqueTriangles;

    if (curr_time < 0.3333)
    {
        if (r == 0)
            return HIPRT_ACCEPT_INTERSECTION;
    }
    else if (curr_time >= 0.3333 && curr_time < 0.66666)
    {
        if (r == 1)
            return HIPRT_ACCEPT_INTERSECTION;
    }
    else
    {
        if (r == 2)
            return HIPRT_ACCEPT_INTERSECTION;
            
    }*/

    return HIPRT_ACCEPT_INTERSECTION;
}


HIPRT_DEVICE bool intersectFunc( uint geomType, uint rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, hiprtHit& hit )
{
	 const uint	index = tableHeader.numGeomTypes * rayType + geomType;
	 const void* data  = tableHeader.funcDataSets[index].intersectFuncData;
     switch ( index )
	 {
     case static_cast<uint>(GEOMETRY_TYPE::TRIANGLE_MESH_DEFORMED):
         return IntersectDeformation(ray, data, payload, hit);
	 default:
	 	return HIPRT_INTERSECTION_MISS;
	 }
	return HIPRT_INTERSECTION_MISS;
}

HIPRT_DEVICE bool filterFunc( uint geomType, uint rayType, const hiprtFuncTableHeader& tableHeader, const hiprtRay& ray, void* payload, const hiprtHit& hit )
{
	 const uint	index = tableHeader.numGeomTypes * rayType + geomType;
	 const void* data  = tableHeader.funcDataSets[index].filterFuncData;
	 switch ( index )
	 {
     case static_cast<uint>(GEOMETRY_TYPE::TRIANGLE_MESH_DEFORMED):
         return AnyhitDeformation(ray, data, payload, hit);
	 default:
	 	return HIPRT_ACCEPT_INTERSECTION;
	 }
	return HIPRT_ACCEPT_INTERSECTION;
}



template <uint32_t Option>
__device__ int3
getColor( hiprtScene scene, const hiprtHit& hit)
{
	return int3{};
}

template <uint32_t Option>
__device__ int3 getColor(hiprtGeometry geometry, const hiprtHit& hit)
{
    return int3{};
}

template <>
__device__ int3 getColor<VisualizeHitDist>(
	hiprtScene scene, const hiprtHit& hit)
{
	float t = hit.t / 15.0f;
	int3  color;
	color.x = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	color.y = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	color.z = hiprt::clamp( static_cast<uint32_t>( t * 255 ), 0, 255 );
	return color;
}

template<>
__device__ int3 getColor<VisualizeHitDist>(hiprtGeometry geometry, const hiprtHit& hit)
{
    float t = hit.t / 50.0f;
    int3 color;
    color.x = hiprt::clamp(static_cast<uint32_t>(t * 255), 0, 255);
    color.y = hiprt::clamp(static_cast<uint32_t>(t * 255), 0, 255);
    color.z = hiprt::clamp(static_cast<uint32_t>(t * 255), 0, 255);
    return color;
}


template <uint32_t Option>
__device__ void PrimaryRayKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	int2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	const Camera&		   camera,
	float				   aoRadius )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	uint32_t seed = tea<16>( x + y * resolution.x, 0 ).x;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	hiprtRay													ray = generateRay( x, y, resolution, camera, seed, false );
	hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
	{
		hiprtHit hit = tr.getNextHit();
		int3	 color{0, 0, 0};
		if ( hit.hasHit() ) color = getColor<VisualizeHitDist>( scene, hit);

		image[index * 4 + 0] = color.x;
		image[index * 4 + 1] = color.y;
		image[index * 4 + 2] = color.z;
		image[index * 4 + 3] = 255;
	}
}

extern "C" __global__ void __launch_bounds__(64) PrimaryRayKernel_RayDist(
	hiprtScene			   scene,
	uint8_t*			   image,
	int2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	Camera				   camera,
	float				   aoRadius )
{
	PrimaryRayKernel<VisualizeHitDist>(
		scene,
		image,
		resolution,
		globalStackBuffer,
		camera,
		aoRadius );
}


extern "C" __global__ void __launch_bounds__(64) PrimaryRaysForGeometryKernel_RayDist(
	hiprtGeometry geometry,
	uint8_t* image,
	int2 resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	const Camera camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    __shared__ uint32_t sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtSharedStackBuffer sharedStackBuffer{SHARED_STACK_SIZE, sharedStackCache};

	Stack stack(globalStackBuffer, sharedStackBuffer);
    InstanceStack instanceStack;

	uint32_t seed = tea<16>(x + y * resolution.x, 0).x;

	hiprtRay ray = generateRay(x, y, resolution, camera, seed, false);
    hiprtGeomCustomTraversalClosestCustomStack<Stack> tr(geometry, ray, stack);

	hiprtHit hit = tr.getNextHit();
    int3 color{0, 0, 0};
    if (hit.hasHit())
        color = getColor<VisualizeHitDist>(geometry, hit);

    image[index * 4 + 0] = color.x;
    image[index * 4 + 1] = color.y;
    image[index * 4 + 2] = color.z;
    image[index * 4 + 3] = 255;

}

/*
* 	const float3 holDir	 = rotate( camera.m_rotation, make_float3( 1.0f, 0.0f, 0.0f ) );
    const float3 upDir	 = rotate( camera.m_rotation, make_float3( 0.0f, -1.0f, 0.0f ) );
    const float3 viewDir = rotate( camera.m_rotation, make_float3( 0.0f, 0.0f, -1.0f ) );
*/
extern "C" __global__ void SimpleMeshIntersectionKernel(hiprtGeometry geom, uint8_t* image, int2 resolution)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    hiprtRay ray;
    const float3 o = {x / static_cast<float>(resolution.x), y / static_cast<float>(resolution.y), -1.0f};
    const float3 d = {0.0f, 0.0f, 1.0f};
    ray.origin = o;
    ray.direction = d;

    hiprtGeomTraversalClosest tr(geom, ray);
    hiprtHit hit = tr.getNextHit();

	uint8_t ix = hit.hasHit() ? (static_cast<float>(x) / resolution.x) * 255 : 0;
    uint8_t iy = hit.hasHit() ? (static_cast<float>(y) / resolution.y) * 255 : 0;

    image[index * 4 + 0] = ix;
    image[index * 4 + 1] = iy;
    image[index * 4 + 2] = 0;
    image[index * 4 + 3] = 255;
}

extern "C" __global__ void SimpleMeshIntersectionKernelCamera(hiprtGeometry geom, uint8_t* image, int2 resolution, const Camera camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;
   
	uint32_t seed = tea<16>(x + y * resolution.x, 0).x;

	hiprtRay ray  = generateRay(x, y, resolution, camera, seed, false);
    
    hiprtGeomTraversalClosest tr(geom, ray);
    hiprtHit hit = tr.getNextHit();

    uint8_t ix = hit.hasHit() ? (static_cast<float>(x) / resolution.x) * 255 : 0;
    uint8_t iy = hit.hasHit() ? (static_cast<float>(y) / resolution.y) * 255 : 0;

    image[index * 4 + 0] = ix;
    image[index * 4 + 1] = iy;
    image[index * 4 + 2] = 0;
    image[index * 4 + 3] = 255;
}

extern "C" __global__ void SimpleMeshIntersectionKernelCameraTx(hiprtGeometry geom, hipTextureObject_t textureObject, int2 resolution, const Camera camera)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    uint32_t seed = tea<16>(x + y * resolution.x, 0).x;

    hiprtRay ray = generateRay(x, y, resolution, camera, seed, false);

    hiprtGeomTraversalClosest tr(geom, ray);
    hiprtHit hit = tr.getNextHit();

    uint8_t ix = hit.hasHit() ? (static_cast<float>(x) / resolution.x) * 255 : 0;
    uint8_t iy = hit.hasHit() ? (static_cast<float>(y) / resolution.y) * 255 : 0;

    /*texture*/

   /* image[index * 4 + 0] = ix;
    image[index * 4 + 1] = iy;
    image[index * 4 + 2] = 0;
    image[index * 4 + 3] = 255;*/
}



extern "C" __global__ void __launch_bounds__( 64 ) AoRayKernel(
	hiprtScene			   scene,
	uint8_t*			   image,
	int2				   resolution,
	hiprtGlobalStackBuffer globalStackBuffer,
	Camera				   camera,
	float				   aoRadius,
	hiprtFuncTable		   table )
{
	const uint32_t x	 = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y	 = blockIdx.y * blockDim.y + threadIdx.y;
	const uint32_t index = x + y * resolution.x;

	constexpr uint32_t Spp		 = 512;
	constexpr uint32_t AoSamples = 32;

	int3   color{};
	float3 diffuseColor = make_float3( 1.0f );
	float  ao			= 0.0f;

	__shared__ uint32_t	   sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
	hiprtSharedStackBuffer sharedStackBuffer{ SHARED_STACK_SIZE, sharedStackCache };

	Stack		  stack( globalStackBuffer, sharedStackBuffer );
	InstanceStack instanceStack;

	

	for ( uint32_t p = 0; p < Spp; p++ )
	{
		uint32_t seed = tea<16>( x + y * resolution.x, p ).x;
		
		hiprtRay													ray = generateRay( x, y, resolution, camera, seed, true );
		hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr( scene, ray, stack, instanceStack );
		{
			hiprtHit hit = tr.getNextHit();

			if ( hit.hasHit() )
			{
				
				const float3 surfacePt = ray.origin + hit.t * ( 1.0f - 1.0e-2f ) * ray.direction;

				float3 Ng = hiprtVectorObjectToWorld( hit.normal, scene, hit.instanceID );
				if ( hiprt::dot( ray.direction, Ng ) > 0.0f ) Ng = -Ng;
				Ng = hiprt::normalize( Ng );

				hiprtRay aoRay;
				aoRay.origin = surfacePt;
				aoRay.maxT	 = aoRadius;
				hiprtHit aoHit;

				for ( uint32_t i = 0; i < AoSamples; i++ )
				{
					aoRay.direction = sampleHemisphereCosine( Ng, seed );
					hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr(
						scene, aoRay, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table );
					aoHit = tr.getNextHit();
					ao += !aoHit.hasHit() ? 1.0f : 0.0f;
				}
				
			}
		}
	}

	ao = ao / ( Spp * AoSamples );

	color.x = ( ao * diffuseColor.x ) * 255;
	color.y = ( ao * diffuseColor.y ) * 255;
	color.z = ( ao * diffuseColor.z ) * 255;

	image[index * 4 + 0] = color.x;
	image[index * 4 + 1] = color.y;
	image[index * 4 + 2] = color.z;
	image[index * 4 + 3] = 255;
}

extern "C" __global__ void __launch_bounds__(64)
    AoRayKernelMotionBlurSlerp(hiprtScene scene, uint8_t* image, int2 resolution, hiprtGlobalStackBuffer globalStackBuffer, Camera camera, float aoRadius, hiprtFuncTable table)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    constexpr uint32_t Spp = 512;
    constexpr uint32_t AoSamples = 64;

     const float3 colors[2] = {{1.0f, 0.0f, 0.5f}, {0.0f, 0.5f, 1.0f}};

    int3 color{};
    float3 diffuseColor = make_float3(1.0f);
    float ao = 0.0f;

    __shared__ uint32_t sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtSharedStackBuffer sharedStackBuffer{SHARED_STACK_SIZE, sharedStackCache};

    Stack stack(globalStackBuffer, sharedStackBuffer);
    InstanceStack instanceStack;
    for (uint32_t p = 0; p < Spp; p++)
    {
        uint32_t seed = tea<16>(x + y * resolution.x, p).x;
        float time = randf(seed);
        hiprtRay ray = generateRay(x, y, resolution, camera, seed, true);
        hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr(scene, ray, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table, 0, time);
        {
            hiprtHit hit = tr.getNextHit();

            if (hit.hasHit())
            {
                
                const float3 surfacePt = ray.origin + hit.t * (1.0f - 1.0e-2f) * ray.direction;
                diffuseColor = colors[hit.instanceID];
                float3 Ng = hiprtVectorObjectToWorld(hit.normal, scene, hit.instanceID);
                if (hiprt::dot(ray.direction, Ng) > 0.0f)
                    Ng = -Ng;
                Ng = hiprt::normalize(Ng);

                hiprtRay aoRay;
                aoRay.origin = surfacePt;
                aoRay.maxT = aoRadius;
                hiprtHit aoHit;

                for (uint32_t i = 0; i < AoSamples; i++)
                {
                    
                    aoRay.direction = sampleHemisphereCosine(Ng, seed);
                    hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr(scene, aoRay, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, table, 0, time);
                    aoHit = tr.getNextHit();
                    ao += !aoHit.hasHit() ? 1.0f : 0.0f;
                    if (aoHit.hasHit())
                    {
                        diffuseColor = colors[aoHit.instanceID];
                    }

                    
                }
            }
        }
    }

    ao = ao / (Spp * AoSamples);
 
    color.x = (ao * diffuseColor.x) * 255;
    color.y = (ao * diffuseColor.y) * 255;
    color.z = (ao * diffuseColor.z) * 255;

    image[index * 4 + 0] = color.x;
    image[index * 4 + 1] = color.y;
    image[index * 4 + 2] = color.z;
    image[index * 4 + 3] = 255;
}


extern "C" __global__ void __launch_bounds__(64)
    MotionBlurrRayKernelSampling(hiprtScene scene, uint8_t* image, int2 resolution, hiprtGlobalStackBuffer globalStackBuffer, Camera camera, float aoRadius, hiprtFuncTable table)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

	// Number of blurred entities in the output image will depend on Samples;
    constexpr uint32_t Samples = 3u;

    const float3 colors[2] = {{1.0f, 0.0f, 0.5f}, {0.0f, 0.5f, 1.0f}};

	
    float3 color{};
    for (uint32_t i = 0; i < Samples; ++i)
    {
        uint32_t seed = tea<16>(x + y * resolution.x, i).x;

        hiprtRay ray = generateRay(x, y, resolution, camera, seed, true);
		  
		const float time = i / static_cast<float>(Samples);

        hiprtSceneTraversalClosest tr(scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time);
      
		hiprtHit hit = tr.getNextHit();
        if (hit.hasHit())
            color += colors[hit.instanceID];
    }

    color = gammaCorrect(color / Samples);

    image[index * 4 + 0] = color.x * 255;
    image[index * 4 + 1] = color.y * 255;
    image[index * 4 + 2] = color.z * 255;
    image[index * 4 + 3] = 255;
}

extern "C" __global__ void __launch_bounds__(64) MotionBlurrRayKernelSamplingDeformation(hiprtScene scene,
                                                                              uint8_t* image,
                                                                              int2 resolution,
                                                                              hiprtGlobalStackBuffer globalStackBuffer,
                                                                              Camera camera,
                                                                              float aoRadius,
                                                                              hiprtFuncTable table,
                                                                              GeometryData* data)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;
   
    // Number of blurred entities in the output image will depend on Samples;
    constexpr uint32_t Samples = 3u;

      __shared__ uint32_t sharedStackCache[SHARED_STACK_SIZE * BLOCK_SIZE];
    hiprtSharedStackBuffer sharedStackBuffer{SHARED_STACK_SIZE, sharedStackCache};

    Stack stack(globalStackBuffer, sharedStackBuffer);
    InstanceStack instanceStack;

    const float3 colors[2] = {{1.0f, 0.0f, 0.5f}, {0.0f, 0.5f, 1.0f}};

    Payload payload{};
    payload.resolution = resolution;

    float3 color{};
    for (uint32_t i = 0; i < Samples; ++i)
    {
        uint32_t seed = tea<16>(x + y * resolution.x, i).x;

        hiprtRay ray = generateRay(x, y, resolution, camera, seed, true);

        const float time = i / static_cast<float>(Samples);
        payload.time = time;
        hiprtSceneTraversalAnyHitCustomStack<Stack, InstanceStack> tr(scene, ray, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, &payload, table, 0, time);
        //hiprtSceneTraversalClosestCustomStack<Stack, InstanceStack> tr(scene, ray, stack, instanceStack, hiprtFullRayMask, hiprtTraversalHintDefault, &payload, table, 0, time);
        hiprtHit hit = tr.getNextHit();

        if (hit.hasHit())
        {
            GeometryData& meshData = data[hit.instanceID];
            uint3& triangle = meshData.indices[hit.primID];

            float beta = hit.uv.x;
            float gamma = hit.uv.y;
            float alpha = 1.f - beta + gamma;

            float3 n0, n1, n2;
            if (meshData.nDeformations > 1)
            {
                //interpolate normals
                n0 = meshData.vertex_normals[triangle.x];
                n1 = meshData.vertex_normals[triangle.y];
                n2 = meshData.vertex_normals[triangle.z];  
            
            }
            else
            {
                n0 = meshData.vertex_normals[triangle.x];
                n1 = meshData.vertex_normals[triangle.y];
                n2 = meshData.vertex_normals[triangle.z];            
            }
                   

            
            
            float3 sn = alpha * n0 + beta * n1 + gamma * n2; // b0 * v0.normal + hp.b1 * v1.normal + hp.b2 * v2.normal;
                        
            sn = hiprt::normalize(hiprtVectorObjectToWorld(sn, scene, hit.instanceID));
            color = 0.5f * sn + make_float3(0.5f);
        }
    }

    color = gammaCorrect(color / Samples);

    image[index * 4 + 0] = color.x * 255;
    image[index * 4 + 1] = color.y * 255;
    image[index * 4 + 2] = color.z * 255;
    image[index * 4 + 3] = 255;
}

extern "C" __global__ void __launch_bounds__(64)
    MotionBlurrRayKernelSlerp(hiprtScene scene, uint8_t* image, int2 resolution, hiprtGlobalStackBuffer globalStackBuffer, Camera camera, float aoRadius, hiprtFuncTable table)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    constexpr uint32_t Samples = 512u;

    const float3 colors[2] = {{1.0f, 0.0f, 0.5f}, {0.0f, 0.5f, 1.0f}};

    float3 color{};
    for (uint32_t i = 0; i < Samples; ++i)
    {
		
        uint32_t seed = tea<16>(x + y * resolution.x, i).x;
        // here we just shot random times and with the results we can get an object smeared in output
        float time = randf(seed);

        hiprtRay ray = generateRay(x, y, resolution, camera, seed, true);

        hiprtSceneTraversalClosest tr(scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time);
  
        hiprtHit hit = tr.getNextHit();
        if (hit.hasHit())
            color += colors[hit.instanceID];
    }

    color = gammaCorrect(color / Samples);

    image[index * 4 + 0] = color.x * 255;
    image[index * 4 + 1] = color.y * 255;
    image[index * 4 + 2] = color.z * 255;
    image[index * 4 + 3] = 255;
}

extern "C" __global__ void __launch_bounds__(64) MotionBlurrRayKernelDeformation(hiprtScene scene,
                                                                                 uint8_t* image,
                                                                                 int2 resolution,
                                                                                 hiprtGlobalStackBuffer globalStackBuffer,
                                                                                 Camera camera,
                                                                                 float aoRadius,
                                                                                 hiprtFuncTable table)
{
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t index = x + y * resolution.x;

    constexpr uint32_t Samples = 512u;

    const float3 colors[2] = {{1.0f, 0.0f, 0.5f}, {0.0f, 0.5f, 1.0f}};

    float3 color{};
    for (uint32_t i = 0; i < Samples; ++i)
    {
        uint32_t seed = tea<16>(x + y * resolution.x, i).x;
        // here we just shot random times and with the results we can get an object smeared in output
        float time = randf(seed);

        hiprtRay ray = generateRay(x, y, resolution, camera, seed, true);

        hiprtSceneTraversalClosest tr(scene, ray, hiprtFullRayMask, hiprtTraversalHintDefault, nullptr, nullptr, 0, time);

        hiprtHit hit = tr.getNextHit();
        if (hit.hasHit())
        {
			//get the values from the given frame
			
        
			
			
			color += colors[hit.instanceID];
        
		}
    }

    color = gammaCorrect(color / Samples);

    image[index * 4 + 0] = color.x * 255;
    image[index * 4 + 1] = color.y * 255;
    image[index * 4 + 2] = color.z * 255;
    image[index * 4 + 3] = 255;

}