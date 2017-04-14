#include "object.h"

__global__ void run_vbo_kernel(float4 *pos, Object* d_objects, float time)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    int object_id = getObjectId(idx, d_objects);
    if(object_id == -1)
        return;
    float4 speed = d_objects[object_id].speed;
    if(idx == d_objects[object_id].start_index){
    	d_objects[object_id].centroid = make_float4(d_objects[object_id].centroid.x + speed.x*time,
    												d_objects[object_id].centroid.y + speed.y*time, 
    	 											d_objects[object_id].centroid.z + speed.z*time, 1.0f); 

    	d_objects[object_id].bV.u_f_r = make_float3(d_objects[object_id].bV.u_f_r.x + time*speed.x, d_objects[object_id].bV.u_f_r.y + time*speed.y, d_objects[object_id].bV.u_f_r.z + time*speed.z);
		d_objects[object_id].bV.u_f_l = make_float3(d_objects[object_id].bV.u_f_l.x + time*speed.x, d_objects[object_id].bV.u_f_l.y + time*speed.y, d_objects[object_id].bV.u_f_l.z + time*speed.z);
		
		d_objects[object_id].bV.u_b_r = make_float3(d_objects[object_id].bV.u_b_r.x + time*speed.x, d_objects[object_id].bV.u_b_r.y + time*speed.y, d_objects[object_id].bV.u_b_r.z + time*speed.z);
		d_objects[object_id].bV.u_b_l = make_float3(d_objects[object_id].bV.u_b_l.x + time*speed.x, d_objects[object_id].bV.u_b_l.y + time*speed.y, d_objects[object_id].bV.u_b_l.z + time*speed.z);
		
		d_objects[object_id].bV.lo_f_r = make_float3(d_objects[object_id].bV.lo_f_r.x + time*speed.x, d_objects[object_id].bV.lo_f_r.y + time*speed.y, d_objects[object_id].bV.lo_f_r.z + time*speed.z);
		d_objects[object_id].bV.lo_f_l = make_float3(d_objects[object_id].bV.lo_f_l.x + time*speed.x, d_objects[object_id].bV.lo_f_l.y + time*speed.y, d_objects[object_id].bV.lo_f_l.z + time*speed.z);
		
		d_objects[object_id].bV.lo_b_r = make_float3(d_objects[object_id].bV.lo_b_r.x + time*speed.x, d_objects[object_id].bV.lo_b_r.y + time*speed.y, d_objects[object_id].bV.lo_b_r.z + time*speed.z);
		d_objects[object_id].bV.lo_b_l = make_float3(d_objects[object_id].bV.lo_b_l.x + time*speed.x, d_objects[object_id].bV.lo_b_l.y + time*speed.y, d_objects[object_id].bV.lo_b_l.z + time*speed.z);
    }
    pos[idx] = make_float4(pos[idx].x + speed.x*time, 
    	pos[idx].y + speed.y*time, 
    	pos[idx].z+speed.z*time,1.0f);
}

// __device__ int hash()
__global__ void find_CellID(Object* d_objects, int *cellids, int* objectIds, float CELL_SIZE){
    int X_SHIFT = 10, Y_SHIFT = 10, Z_SHIFT = 10;
    int idx = blockIdx.x*blockDim.x + threadIdx.x;

	cellids[8*idx] = 1 << 31 + ((int)(d_objects[idx].centroid.x / CELL_SIZE) << X_SHIFT + 
							(int)(d_objects[idx].centroid.y / CELL_SIZE) << Y_SHIFT +
							(int)(d_objects[idx].centroid.z / CELL_SIZE) << Z_SHIFT);  
	objectIds[8*idx] = idx; 

	int temp = ((int)(d_objects[idx].bV.u_f_r.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.u_f_r.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.u_f_r.z / CELL_SIZE) << Z_SHIFT);

	cellids[8*idx+1] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+1] = idx;

	temp = ((int)(d_objects[idx].bV.u_f_l.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.u_f_l.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.u_f_l.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+2] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+2] = idx;

	temp = ((int)(d_objects[idx].bV.u_b_r.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.u_b_r.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.u_b_r.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+3] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+3] = idx;

	temp = ((int)(d_objects[idx].bV.u_b_l.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.u_b_l.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.u_b_l.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+4] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+4] = idx;

	temp = ((int)(d_objects[idx].bV.lo_f_r.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.lo_f_r.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.lo_f_r.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+5] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+5] = idx;

	temp = ((int)(d_objects[idx].bV.lo_f_l.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.lo_f_l.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.lo_f_l.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+6] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+6] = idx;

	temp = ((int)(d_objects[idx].bV.lo_b_r.x / CELL_SIZE) << X_SHIFT + 
				(int)(d_objects[idx].bV.lo_b_r.y / CELL_SIZE) << Y_SHIFT +
				(int)(d_objects[idx].bV.lo_b_r.z / CELL_SIZE) << Z_SHIFT);
	cellids[8*idx+7] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	objectIds[8*idx+7] = idx;

	// temp = ((int)(d_objects[idx].bV.lo_b_l.x / CELL_SIZE) << X_SHIFT + 
	// 			(int)(d_objects[idx].bV.lo_b_l.y / CELL_SIZE) << Y_SHIFT +
	// 			(int)(d_objects[idx].bV.lo_b_l.z / CELL_SIZE) << Z_SHIFT);
	// cellids[8*idx+8] = 0 << 31 + temp*(~(temp ^ (cellids[8*idx] & 0x7FFFFFFF))&1);
	// objectIds[8*idx+8] = idx;
}

// float getMaximumBoundingBox(std::vector <std::vector <glm::vec3> >  concatenated_vectices){
//     float maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX, maxZ = INT_MIN, minZ = INT_MAX;
//     float maxBox = INT_MIN;
//     for (int i = 0; i < concatenated_vectices.size(); ++i)
//     {
//         maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX, maxZ = INT_MIN, minZ = INT_MAX;
//         for (int j = 0; j < concatenated_vectices[i].size(); ++j)
//         {
//             maxX = max(maxX, concatenated_vectices[i][j].x);
//             minX = min(minX, concatenated_vectices[i][j].x);
//             maxZ = max(maxZ, concatenated_vectices[i][j].z);
//             minZ = min(minZ, concatenated_vectices[i][j].z);
//             maxY = max(maxY, concatenated_vectices[i][j].y);
//             minY = min(minY, concatenated_vectices[i][j].y);
//         }
//         maxBox = max(maxBox, max(maxX - minX, max(maxY - minY, maxZ - minZ)));
//     }
//     return maxBox;
// }
