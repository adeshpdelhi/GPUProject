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
__device__ float3 averagePoint(float4 *vertices, int start_index, int size){
	float3 avg = make_float3(0.0, 0.0, 0.0);
	for( int i = start_index; i < start_index+size; i++){
		avg.x += vertices[i].x;
		avg.y += vertices[i].y;
		avg.z += vertices[i].z;
	}
	avg.x /= size;
	avg.y /= size;
	avg.z /= size;
	return avg;	
}

__device__ float3 getFarthestPointInDirection(float4 * vertices, int start_index, int size, float3 direction){
	float max = INT_MIN;
	int pos = -1;
	for (int i = start_index; i < start_index+size; i++){
		float x = dot(make_float3(vertices[i].x, vertices[i].y, vertices[i].z), direction);
		if(x > max){
			max = x;
			pos = i;
		}
	}
	return make_float3(vertices[pos].x ,vertices[pos].y, vertices[pos].z);
}
// take vertices form d_ptr (run_vbo_kernel)
__device__ float3 support(float4 *vertices, int start_index_1, int size1, int start_index_2, int size2, float3 direction){
	float3 pt1 = getFarthestPointInDirection(vertices, start_index_1, size1, direction);
	float3 pt2 = getFarthestPointInDirection(vertices, start_index_2, size2, negate(direction));
	return subtract(pt1, pt2);
}

__device__ float3 tripleProduct(float3 a , float3 b, float3 c){
	return subtract(scale(b, dot(c,a)), scale(c,dot(a,b)));
}

__device__ float3 crossProduct(float3 a, float3 b){
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

__device__ bool containsOrigin(float3 simplex[], float3 *direction){
	
	// normal perpendicular to and outside each face: 

	// simplex[0] == D
	// simplex[1] == C
	// simplex[2] == B
	// simplex[3] == A
	// ADxAB ---> (ADXAB).AC < 0
	float3 ABD_norm = crossProduct(subtract(simplex[0],simplex[3]),subtract(simplex[2], simplex[3]));
	if(dot(ABD_norm, subtract(simplex[1],simplex[3])) >= 0){
		// printf("negated perp\n");
		ABD_norm = negate(ABD_norm);
	}
	// ACxAD ---> (ACxAD).AB < 0
	float3 ACD_norm = crossProduct(subtract(simplex[1],simplex[3]),subtract(simplex[0], simplex[3]));
	if(dot(ACD_norm, subtract(simplex[2],simplex[3])) >= 0){
		// printf("negated perp\n");
		ACD_norm = negate(ACD_norm);
	}
	// ABxAC ---> (ABxAC).AD < 0
	float3 ABC_norm = crossProduct(subtract(simplex[2],simplex[3]),subtract(simplex[1], simplex[3]));
	if(dot(ABC_norm, subtract(simplex[0],simplex[3])) >= 0){
		// printf("negated perp\n");
		ABC_norm = negate(ABC_norm);
	}
	// AO == negate(A)
	float3 negated_A = negate(simplex[3]);
	if(dot(ABD_norm, negated_A) >= 0){
		// printf("1...........\n");
		
		// points in plane A B D will remain - D = D, C = B, B = A 
		simplex[1] = simplex[2]; // C = B
		simplex[2] = simplex[3];  // B = A
		// printf("recomputing direction\n");
		
		*(direction) = ABD_norm;
	}else if(dot(ACD_norm, negated_A) >= 0){
		// printf("2...........\n");

		// points in plane ACD will remain - D = D , C = C , B = A
		simplex[2] = simplex[3]; // B = A
		// printf("recomputing direction\n");
		
		*(direction) = ACD_norm;
	}else if(dot(ABC_norm, negated_A) >= 0){
		// printf("3...........\n");

		// point in plane ABC will remain -  D = C, C = B , B = A
		simplex[0] = simplex[1]; // D = C
		simplex[1] = simplex[2]; // C = B
		simplex[2] = simplex[3]; // B = A
		// printf("recomputing direction\n");

		*(direction) = ABC_norm;
	}else{
		//collision
		// printf("collision detected...........\n");
		return true;
	}
	// printf("collision not detected yet\n");
	return false;
}
// __device__ float3 normalize(float3 a){
// 	float norm = sqrt(a.x*a.x + a.y*a.y + a.z*a.z);
// 	return make_float3(a.x/norm, a.y/norm, a.z/norm);
// }

__device__ bool gjk(float4 *all_vertices, Object *objects,  int objectId1, int objectId2){

	int size1 = objects[objectId1].n_vertices;
	int start_index_1 = objects[objectId1].start_index;

	int size2 = objects[objectId2].n_vertices;
	int start_index_2 = objects[objectId2].start_index;

	// printf("vertiex 02 : %f %f %f\n",all_vertices[start_index_2].x, all_vertices[start_index_2].y, all_vertices[start_index_2].z );
	// printf("vertiex 12 : %f %f %f\n",all_vertices[start_index_2+1].x, all_vertices[start_index_2+1].y, all_vertices[start_index_2+1].z );
	// printf("vertiex size2-1 : %f %f %f\n",all_vertices[start_index_2+size2-1].x, all_vertices[start_index_2+size2-1].y, all_vertices[start_index_2+size2-1].z );
	
	float3 simplex[4];

	float3 position1 = averagePoint(all_vertices, start_index_1, size1);
	// printf("position 1: %f %f %f \n", position1.x, position1.y, position1.z );
	float3 position2 = averagePoint(all_vertices, start_index_2, size2);
	// printf("position 2: %f %f %f \n", position2.x, position2.y, position2.z );

	float3 direction =  subtract(position1, position2);
	// direction = normalize(direction);
	// printf("initial direction to find simplex 0 and 1: %f %f %f \n", direction.x, direction.y, direction.z );
	if(direction.x == 0.0 && direction.y == 0 && direction.z == 0)
		direction.y = 1.0f;

	// B
	simplex[0] = support(all_vertices, start_index_1, size1, start_index_2, size2, direction);
	
	// A
	simplex[1] = support(all_vertices, start_index_1, size1, start_index_2, size2, negate(direction));

	// (ABxAo)xAB ; AB = B-A, AO = negateA
	direction = tripleProduct(subtract(simplex[0],simplex[1]), negate(simplex[1]),subtract(simplex[0],simplex[1]));
	// check if length of line is 0, ie direction obtained is (0,0,0), which will happen when AB and AO are parallel
	//  in this case take perp direction of line (perp = -y, x)
	// if()

	// direction = normalize(direction);
	// printf("direction (ABxAO)xAB[2]: %f %f %f\n", direction.x,direction.y, direction.z);

	//  A_new ;  C = simplex[0] , B = simplex[1], A = simplex[2]
	simplex[2] = support(all_vertices, start_index_1, size1, start_index_2, size2, direction);

	// ACxAB ; direction perpendicular to plane containing triangle
	direction = crossProduct(subtract(simplex[0],simplex[2]), subtract(simplex[1], simplex[2]));
	if(dot(direction, negate(simplex[2])) < 0 ){
		// printf("negating ACxAB\n");
		direction = negate(direction);
	}
	// direction = normalize(direction);
	while(1){
		// printf("direction [3] ABxAC: %f %f %f\n", direction.x,direction.y, direction.z);
		simplex[3] = support(all_vertices, start_index_1, size1, start_index_2, size2, direction);
		// printf("simplex[0]: %f %f %f\n", simplex[0].x,simplex[0].y,simplex[0].z);
		// printf("simplex[1]: %f %f %f\n", simplex[1].x,simplex[1].y,simplex[1].z);
		// printf("simplex[2]: %f %f %f\n", simplex[2].x, simplex[2].y, simplex[2].z);
		// printf("simplex[3]: %f %f %f\n", simplex[3].x, simplex[3].y, simplex[3].z);
				
		if(dot(simplex[3], direction ) <= 0 ){
			// printf("whaaaaaaaaaat\n");
			return false;
		}
		else{
			if(containsOrigin(simplex, &direction)){
				// printf("collision detected :D\n");
				return true;
			}
		}
		// ABxAC ; AB - B-A , AC - C-A
		// direction = crossProduct(subtract(simplex[1],simplex[2]), subtract(simplex[0],simplex[2]));
		// direction = normalize(direction);
	}

}