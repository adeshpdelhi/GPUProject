//Referenced from https://github.com/kroitor/gjk.c/blob/master/gjk.c

// #include "gjk.h"
__device__ float3 subtract(float3 a, float3 b ){
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ float3 negate(float3 a){
	return make_float3(-a.x, -a.y, -a.z);
}
__device__ float dot(float3 a, float3 b){
	return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ float3 scale(float3 a, float c){
	return make_float3(a.x * c, a.y * c, a.z * c);
}
__device__ float3 averagePoint(float4 *vertices, int size){
	float3 avg = make_float3(0.0, 0.0, 0.0);
	for( int i = 0; i < size; i++){
		avg.x += vertices[i].x;
		avg.y += vertices[i].y;
		avg.z += vertices[i].z;
	}
	avg.x /= size;
	avg.y /= size;
	avg.z /= size;
	return avg;	
}

__device__ float3 getFarthestPointInDirection(float4 * vertices, int size, float3 direction){
	float max = INT_MIN;
	int pos = -1;
	for (int i = 0; i < size; i++){
		float x = dot(make_float3(vertices[i].x, vertices[i].y, vertices[i].z), direction);
		if(x > max){
			max = x;
			pos = i;
		}
	}
	return make_float3(vertices[pos].x ,vertices[pos].y, vertices[pos].z);
}
// take vertices form d_ptr (run_vbo_kernel)
__device__ float3 support(float4 *vertices1, int size1, float4 *vertices2, int size2, float3 direction){
	float3 pt1 = getFarthestPointInDirection(vertices1, size1, direction);
	float3 pt2 = getFarthestPointInDirection(vertices2, size2, negate(direction));
	return subtract(pt1, pt2);

}

__device__ float3 tripleProduct(float3 a , float3 b, float3 c){
	return subtract(scale(b, dot(c,a)), scale(a,dot(c,b)));
}

__device__ float3 crossProduct(float3 a, float3 b){
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ bool containsOrigin(float3 simplex[], float3 *direction){
	// ADxAB
	float3 ABD_norm = crossProduct(subtract(simplex[0],simplex[3]),subtract(simplex[2], simplex[3]));
	// ADxAC
	float3 ACD_norm = crossProduct(subtract(simplex[0],simplex[3]),subtract(simplex[1], simplex[3]));
	// ACxAB
	float3 ABC_norm = crossProduct(subtract(simplex[1],simplex[3]),subtract(simplex[2], simplex[3]));
	// Ao == negate(A)
	float3 negated_A = negate(simplex[3]);
	if(dot(ABD_norm, negated_A) >= 0){
		simplex[1] = simplex[2];
		simplex[2] = simplex[3];
		*(direction) = ABD_norm;
	}else if(dot(ACD_norm, negated_A) >= 0){
		simplex[2] = simplex[3];
		*(direction) = ACD_norm;
	}else if(dot(ABC_norm, negated_A) >= 0){
		simplex[0] = simplex[1];
		simplex[1] = simplex[2];
		simplex[2] = simplex[3];
		*(direction) = ABC_norm;
	}else{
		return true;
	}
	return false;

}

__global__ void gjk(float4 *all_vertices, Object *objects,  int objectId1, int objectId2, bool *result){

	printf("working ...\n");	
	float4 *vertices1 = &all_vertices[objects[objectId1].start_index];
	float size1 = objects[objectId1].n_vertices;
	float4 *vertices2 = &all_vertices[objects[objectId2].start_index];
	float size2 = objects[objectId2].n_vertices;

	float3 simplex[4];

	float3 position1 = averagePoint(vertices1, size1);
	float3 position2 = averagePoint(vertices2,size2);

	float3 direction =  subtract(position1, position2);
	if(direction.x == 0.0 && direction.y == 0 && direction.z == 0)
		direction.y = 1.0f;

	// B
	simplex[0] = support(vertices1, size1, vertices2, size2, direction);
	// A
	simplex[1] = support(vertices1, size1, vertices2, size2, negate(direction));
	// (ABxAo)xAB
	direction = tripleProduct(subtract(simplex[0],simplex[1]), negate(simplex[1]),subtract(simplex[0],simplex[1]));
	//  A_new
	simplex[2] = support(vertices1, size1, vertices2, size2, direction);
	// ACxAB
	direction = crossProduct(subtract(simplex[0],simplex[2]), subtract(simplex[1], simplex[2]));

	while(1){
		simplex[3] = support(vertices1, size1, vertices2, size2, direction);
		if(dot(simplex[3], direction ) <= 0 ){
			*(result) =  false;
			return;
		}
		else{
			if(containsOrigin(simplex, &direction)){
				*(result) = true;
				return;
			}
		}
		// simplex[0] = simplex[1];
		// simplex[1] = simplex[2];
		// simplex[2] = simplex[3];
		direction = crossProduct(subtract(simplex[2],simplex[1]), subtract(simplex[2],simplex[0]));
	}

}