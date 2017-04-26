#include <stdio.h>
#include <ctime>
// #include "object.cu"
#include <helper_cuda.h>
// #include "gjk.cu"

#define BLOCK_DIM_PCS 192
#define GRID_DIM_PCS 16

__global__ void createPCSAndCallNarrowPhase(int * d_cellID, int *d_objectID, int partition_size, int total_size, float4 *pos, Object *objects){
	/* Both start and end are inclusive*/

	int thread_start = blockIdx.x*blockDim.x*partition_size + threadIdx.x * partition_size;

	if(thread_start >= total_size)
		return;
	// printf("blockIdx: %d threadIdx: %d thread_start: %d\n", blockIdx.x, threadIdx.x, thread_start);
	bool flag_found_first_transition = false;
	int index_first_transition = -1; //Donotes the first index where the new cell ID is present

	if(thread_start == 0){
		flag_found_first_transition = true;
		index_first_transition = 0;
		// printf("Thread start is zero!\n");
	}
	else
	{
		for(int i = thread_start; i <= thread_start + partition_size; i++){
			if( i >= total_size - 1 )
				continue;
			
			if( (d_cellID[i + 1]>>1) != (d_cellID[i]>>1) ){
				/*First transition detected*/
				flag_found_first_transition = true;
				index_first_transition = i + 1;
			}
		}
	}
	if(flag_found_first_transition == false){
		// printf("First transition not found for blockIdx: %d threadIdx: %d! Safely exiting\n", blockIdx.x, threadIdx.x);
		return;
	}
	
	int index_second_transition = -1; //Donotes the last index where the new cell ID is present
	bool flag_found_second_transition = false;

	int i;
	for (i = index_first_transition; (flag_found_second_transition == false) && (i <= total_size - 2); i++)
	{	
		if((d_cellID[i + 1]>>1) != (d_cellID[i]>>1) ){
				/*Second transition detected*/
				flag_found_second_transition = true;
				index_second_transition = i;
			}
	}
	if(flag_found_second_transition == false && i == total_size - 1)
	{
		flag_found_second_transition = true;
		index_second_transition = total_size - 1;
	}
	if(flag_found_second_transition == false){
		printf("xxxxxxxxxxxxxxxxx ERROR. Second transition not found for blockIdx: %d threadIdx: %d! xxxxxxxxxxxxxxxxx\n", blockIdx.x, threadIdx.x);
		return;
	}
	// printf("BlockId: %d ThreadId: %d ThreadStart: %d First Transition Index %d Second transition index %d\n",blockIdx.x, threadIdx.x,thread_start, index_first_transition, index_second_transition);
	for(int i = index_first_transition; i <= index_second_transition; i++){
		if(d_cellID[i] % 2 == 0){
			/* Home cell*/
			for (int j = i + 1; j <= index_second_transition; j++){
				if(d_cellID[j] % 2 == 1 && d_objectID[i] != d_objectID[j]){
					/* Non home cell found*/
					bool result = gjk(pos, objects, d_objectID[i], d_objectID[j]);
				    printf("result gjk : (%d, %d): %d\n", d_objectID[i], d_objectID[j], result );

				}
			}
		}
		else
		{	/*Non home cell found. All subsequent cells are non home cells. Exit.*/
			// printf("Non home cell exit\n");
			break;
		}
	}

}

// struct pair{
// 	int cellID, objectID;
// };

// int comparator(const void *a, const void *b ){
// 	return (*(pair *)a).cellID - (*(pair*)b).cellID;
// }


// int main(int argc, char const *argv[])
// {
// 	srand(time(NULL));
// 	int OBJECT_COUNT = 3;
// 	int ARRAY_SIZE = 8*OBJECT_COUNT;
// 	int *cellID = (int*) malloc(ARRAY_SIZE*sizeof(int));
// 	int *objectID = (int*) malloc(ARRAY_SIZE*sizeof(int));
// 	if(cellID == NULL || objectID == NULL)
// 	{
// 		printf("Cannot allocate host memory for pcs checking. Aborting!\n");
// 		exit(1);
// 	}
// 	for (int i = 0; i < ARRAY_SIZE; ++i)
// 	{
// 		cellID[i] = rand()%50;
// 		objectID[i] = rand();
// 	}

// 	struct pair *pairs = (struct pair*)malloc(ARRAY_SIZE * sizeof(struct pair));
// 	if(pairs == NULL)
// 	{
// 		printf("Cannot allocate host memory for testing in sorting. Aborting!\n");
// 		exit(1);
// 	}
// 	for (int i = 0; i < ARRAY_SIZE; ++i)
// 	{
// 		pairs[i].cellID = cellID[i];
// 		pairs[i].objectID = objectID[i];
// 	}

// 	qsort(pairs, ARRAY_SIZE, sizeof(pair), comparator);

// 	for (int i = 0; i < ARRAY_SIZE; ++i)
// 	{
// 		cellID[i] = pairs[i].cellID;
// 		objectID[i] = pairs[i].objectID;
// 	}

// 	// for (int i = 0; i < ARRAY_SIZE; ++i)
// 	// {
// 	// 	printf("%d (%d, %d, %d),\n", i, cellID[i]>>1,objectID[i], cellID[i]%2);
// 	// }

// 	// printf("\n");

// 	int * d_cellID, *d_objectID;
// 	checkCudaErrors(cudaMalloc(&d_cellID, ARRAY_SIZE*sizeof(int)));
// 	checkCudaErrors(cudaMalloc(&d_objectID, ARRAY_SIZE*sizeof(int)));
// 	checkCudaErrors(cudaMemcpy(d_cellID, cellID, ARRAY_SIZE*sizeof(int),cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_objectID, objectID, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));
	
// 	int n_blocks = GRID_DIM_PCS, n_threads_per_block = BLOCK_DIM_PCS;
//     int n_threads = n_blocks * n_threads_per_block;
//     dim3 grid2(n_blocks);
//     dim3 block2(n_threads_per_block);
//     int partition_size = ceil(float(8*OBJECT_COUNT)/n_threads);
//     printf("partition_size: %d\n",partition_size );
//     createPCSAndCallNarrowPhase<<<grid2, block2>>>(d_cellID, d_objectID, partition_size, 8*OBJECT_COUNT);

// 	return 0;
// }