#include <stdio.h>
// #include <ctime>
#include "sort.cu"
#include "object.cu"
#include <helper_cuda.h>
#include "gjk.cu"
#include "broadphase.cu"
#include "pcs.cu"
#include "constants.h"
int main(int argc, char const *argv[])
{

	// int NUM_TEMPLATES = 2;
	assert(OBJECT_COUNT > 0);
    for(int i = 0; i < NUM_TEMPLATES; i++){
        std::vector<glm::vec3> tempvertices;
        std::vector<unsigned int> tempIndices;
        bool res = loadOBJ(std::string("./Objects/"+objectsToLoad[i]+".obj").c_str(), tempvertices, tempIndices);
        assert(res);
        templates.insert(Template(tempvertices, tempIndices,objectsToLoad[i],i));
        // printf("ith centroid:  [ %f %f %f]\n", templates.get_ith_template(i).getCentroid().x,templates.get_ith_template(i).getCentroid().y,templates.get_ith_template(i).getCentroid().z);
    }
    
    CELL_SIZE = 1.5*(templates.getMaximumBoundingBox());
    printf("CELLSIZE: %f\n", CELL_SIZE);
    for (int i = 0; i < OBJECT_COUNT; ++i)
    {
        OBJECTS.insert(i%2,i);        
    }
    printf("Test: Size of vertices = %d\n", OBJECTS.vertices.size());
    printf("Test: Size of mappings = %d\n", OBJECTS.mappings.size());
    float time = 0.1;

    float4* pos;
    checkCudaErrors(cudaMalloc(&pos, OBJECTS.vertices.size()*sizeof(float4)));
    checkCudaErrors(cudaMemcpy(pos, &OBJECTS.vertices[0], OBJECTS.vertices.size()*sizeof(float4), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&D_OBJECTS, sizeof(Object)*OBJECT_COUNT));
    checkCudaErrors(cudaMemcpy(D_OBJECTS, OBJECTS.objs, sizeof(Object)*OBJECT_COUNT, cudaMemcpyHostToDevice));
    
    checkCudaErrors(cudaMalloc((void**)&D_OBJ_IDS, sizeof(int)*OBJECTS.vertices.size()));
    checkCudaErrors(cudaMemcpy(D_OBJ_IDS, &OBJECTS.Obj_IDS[0], sizeof(int)*OBJECTS.vertices.size(), cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&D_CELLIDS, sizeof(int)*8*OBJECT_COUNT));
    checkCudaErrors(cudaMalloc(&D_OBJECT_IDS, sizeof(int)*8*OBJECT_COUNT));

    int gridDim = ceil((float)OBJECT_COUNT/BLOCK_DIM);
    dim3 grid(gridDim,1);
    dim3 block(BLOCK_DIM,1);
    printf("grid: %d block: %d\n", grid.x, block.x);
    find_CellID<<< grid, block>>>(D_OBJECTS, D_CELLIDS, D_OBJECT_IDS, CELL_SIZE, OBJECT_COUNT);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("done find_CellID\n");

	sort(D_CELLIDS, D_OBJECT_IDS, 8*OBJECT_COUNT);
	int n_blocks = 16, n_threads_per_block = BLOCK_DIM_PCS;
    int n_threads = n_blocks * n_threads_per_block;
    dim3 grid2(n_blocks);
    dim3 block2(n_threads_per_block);
    int partition_size = ceil(float(8*OBJECT_COUNT)/n_threads);
    printf("partition_size: %d\n",partition_size );
    createPCSAndCallNarrowPhase<<<grid2, block2>>>(D_CELLIDS, D_OBJECT_IDS, partition_size, 8*OBJECT_COUNT, pos, D_OBJECTS);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("done createPCSAndCallNarrowPhase\n");


    int n_vertices = OBJECTS.vertices.size();
    gridDim = ceil((float)n_vertices/BLOCK_DIM);
    dim3 grid1(gridDim,1);
    dim3 block1(BLOCK_DIM,1);
    run_vbo_kernel<<< grid1, block1>>>(pos, D_OBJECTS, time, D_OBJ_IDS, n_vertices);
    printf("done run_vbo_kernel\n");

    time = time + 0.001;
}