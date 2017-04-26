#include <stdio.h>
#include <ctime>
#include <helper_cuda.h>

#define BLOCK_DIM_PCS 10
#define GRID_DIM_PCS 2

__global__ void createPCSAndCallNarrowPhase(int * d_cellID, int *d_objectID, int partition_size, int total_size){
	int block_start = blockIdx.x * blockDim.x;
	int block_end = (blockIdx.x + 1) * blockDim.x - 1;
	/* Both start and end are inclusive*/

	int thread_start = block_start + threadIdx.x * partition_size;
	__shared__ int shared_d_cellID[BLOCK_DIM_PCS];
	__shared__ int shared_d_objectID[BLOCK_DIM_PCS];
	if(thread_start >= total_size)
		return;
	for (int i = 0; i < partition_size; ++i)
	{
		if( thread_start + i >= total_size )
			continue;
		shared_d_cellID[threadIdx.x * partition_size + i] = d_cellID[thread_start + i];
		shared_d_objectID[threadIdx.x * partition_size + i] = d_objectID[thread_start + i];
	}
	__syncthreads();
	bool flag_found_first_transition = false;
	int index_first_transition = -1; //Donotes the first index where the new cell ID is present
	int local_index, global_index;

	if(thread_start == 0){
		flag_found_first_transition = true;
		index_first_transition = 0;
	}
	else
	{
		for(int i = 0; i < partition_size; i++){
			local_index = threadIdx.x * partition_size + i;
			global_index = thread_start + i;
			if( global_index >= total_size )
				continue;
			
			if( global_index == block_end ){
				if((d_cellID[global_index + 1]>>1) != (shared_d_cellID[local_index]>>1) ){
					/*First transition detected*/
					flag_found_first_transition = true;
					index_first_transition = global_index + 1;
				}
			}
			if( (shared_d_cellID[local_index + 1]>>1) != (shared_d_cellID[local_index]>>1) ){
				/*First transition detected*/
				flag_found_first_transition = true;
				index_first_transition = global_index + 1;
			}
		}
	}
	if(flag_found_first_transition == false){
		printf("First transition not found for threadIdx: %d!. Exiting\n", threadIdx.x);
		return;
	}
	int index_second_transition = -1; //Donotes the last index where the new cell ID is present
	int i = index_first_transition;
	bool flag_found_second_transition = false;
	local_index = threadIdx.x * partition_size + i;
	global_index = thread_start + i;
	while(flag_found_second_transition == false && global_index < total_size){
		
		if(global_index <= block_end){
			if( (shared_d_cellID[local_index + 1]>>1) != (shared_d_cellID[local_index]>>1) ){
				/*Second transition detected*/
				flag_found_second_transition = true;
				index_second_transition = global_index;
			}
		}
		else
		{
			if((d_cellID[global_index + 1]>>1) != (d_cellID[global_index]>>1) ){
					/*Second transition detected*/
					flag_found_first_transition = true;
					index_first_transition = global_index;
				}
		}
		i++;
		local_index = threadIdx.x * partition_size + i;
		global_index = thread_start + i;
	}
	if(flag_found_second_transition == false && i == total_size)
	{
		flag_found_second_transition = true;
		index_second_transition = total_size - 1;
	}
	if(flag_found_second_transition == false){
		printf("Error. Second transition not found for threadIdx: %d!\n", threadIdx.x);
		return;
	}
	printf("BlockId: %d ThreadId: %d ThreadStart: %d total_size: %d First Transition Index %d Second transition index %d\n",blockIdx.x, threadIdx.x,thread_start, total_size, index_first_transition, index_second_transition);

}

struct pair{
	int cellID, objectID;
};

int comparator(const void *a, const void *b ){
	return (*(pair *)a).cellID - (*(pair*)b).cellID;
}


int main(int argc, char const *argv[])
{
	srand(time(NULL));
	int OBJECT_COUNT = 3;
	int ARRAY_SIZE = 8*OBJECT_COUNT;
	int *cellID = (int*) malloc(ARRAY_SIZE*sizeof(int));
	int *objectID = (int*) malloc(ARRAY_SIZE*sizeof(int));
	if(cellID == NULL || objectID == NULL)
	{
		printf("Cannot allocate host memory for pcs checking. Aborting!\n");
		exit(1);
	}
	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		cellID[i] = rand()%5;
		objectID[i] = rand();
	}

	struct pair *pairs = (struct pair*)malloc(ARRAY_SIZE * sizeof(struct pair));
	if(pairs == NULL)
	{
		printf("Cannot allocate host memory for testing in sorting. Aborting!\n");
		exit(1);
	}
	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		pairs[i].cellID = cellID[i];
		pairs[i].objectID = objectID[i];
	}

	qsort(pairs, ARRAY_SIZE, sizeof(pair), comparator);

	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		cellID[i] = pairs[i].cellID;
		objectID[i] = pairs[i].objectID;
	}

	for (int i = 0; i < ARRAY_SIZE; ++i)
	{
		printf("(%d, %d, %d),\n", cellID[i]>>1,objectID[i], cellID[i]%2);
	}

	printf("\n");

	int * d_cellID, *d_objectID;
	checkCudaErrors(cudaMalloc(&d_cellID, ARRAY_SIZE*sizeof(int)));
	checkCudaErrors(cudaMalloc(&d_objectID, ARRAY_SIZE*sizeof(int)));
	checkCudaErrors(cudaMemcpy(d_cellID, cellID, ARRAY_SIZE*sizeof(int),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_objectID, objectID, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));
	
	int n_blocks = GRID_DIM_PCS, n_threads_per_block = BLOCK_DIM_PCS;
    int n_threads = n_blocks * n_threads_per_block;
    dim3 grid2(n_blocks);
    dim3 block2(n_threads_per_block);
    int partition_size = ceil(float(8*OBJECT_COUNT)/n_threads);
    printf("partition_size: %d\n",partition_size );
    createPCSAndCallNarrowPhase<<<grid2, block2>>>(d_cellID, d_objectID, partition_size, 8*OBJECT_COUNT);

	return 0;
}