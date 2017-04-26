#include <stdio.h>
#include <ctime>
#include <climits>
#include <helper_cuda.h>
#include <cstdlib>

#define R 16
#define L 8 //Don't increase L beyond 8
#define MAX_GRID_SIZE 2147483646
#define BLOCK_DIM_SORT 192
#define NUMBER_OF_GROUPS_PER_BLOCK 12
#define NUM_RADICES (1<<L)
// #define ARRAY_SIZE 307200
#define NUM_BLOCKS 16 //ceil((float)size/BLOCK_DIM_SORT) //If the size is 3072 = 192*16
// #define ARRAY_SIZE BLOCK_DIM_SORT * NUM_BLOCKS
// #define NUM_THREADS (BLOCK_DIM_SORT * NUM_BLOCKS)
#define NUM_GROUPS (NUMBER_OF_GROUPS_PER_BLOCK * NUM_BLOCKS)
#define NUM_RADICES_PER_BLOCK 16 // NUM_RADICES/NUM_BLOCKS =  256/16 = 16
#define MAX_PASSES 4 //Don't change this

__device__ __host__ int getAddress(int a, int b, int c){
    //d1: radices
    //d2: thread blocks
    //d3: thread groups
    //a: radix, b: thread block, c: thread group

    // int d1 = NUM_RADICES;
	int d2 = NUM_BLOCKS;
	int d3 = NUMBER_OF_GROUPS_PER_BLOCK;
	return (d2*d3*a + d3*b + c);
    //http://stackoverflow.com/questions/789913/array-offset-calculations-in-multi-dimensional-array-column-vs-row-major
}

void __global__ phase_1_kernel(int *d_cellID, int *d_objectID, int size, int partition_size, int Num_Elements_Per_Group, int pass, int * d_counters){
	__shared__ int shared_counters[NUMBER_OF_GROUPS_PER_BLOCK][NUM_RADICES];
	if(threadIdx.x%R==0){
		for (int i = 0; i < NUM_RADICES; ++i){
			shared_counters[threadIdx.x/R][i] = 0;
		}
	}
	__syncthreads();
	int firstCellID =(blockIdx.x*NUMBER_OF_GROUPS_PER_BLOCK + threadIdx.x/R)*Num_Elements_Per_Group + threadIdx.x%R;
	unsigned int mask = 0;
	for (int i = 0; i < L; ++i)
		mask = (mask<<1) | 1;
	mask = mask << (pass*L);
	for (int i = firstCellID; i < firstCellID + R*partition_size && i<size; i+=R)
	{
		unsigned int masked_number = d_cellID[i] & (mask);
		masked_number = masked_number >> (L*pass);
		atomicInc((unsigned int*)&shared_counters[threadIdx.x/R][masked_number], INT_MAX);
	}
	__syncthreads();
	if(threadIdx.x%R == 0){
		for (int i = 0; i < NUM_RADICES; ++i)
		{
			int lk = getAddress(i,blockIdx.x,threadIdx.x/R);
			d_counters[getAddress(i,blockIdx.x,threadIdx.x/R)] = shared_counters[threadIdx.x/R][i];
		}
	}

}

void launch_kernel_phase_1(int* d_cellID, int * d_objectID, int size, int pass, int* d_counters){
    //keep launch config of all kernels same, except kernel 2
	dim3 grid(NUM_BLOCKS);
	dim3 block(BLOCK_DIM_SORT);
	int partition_size = ceil((float)size/(grid.x*block.x));
	int Num_Elements_Per_Group = R*partition_size;
	phase_1_kernel <<<grid, block>>>(d_cellID, d_objectID, size, partition_size, Num_Elements_Per_Group, pass, d_counters);
}

void __global__ phase_2_kernel(int *d_cellID, int *d_objectID, int size, int partition_size, int Num_Elements_Per_Group, int pass, int * d_counters, int *d_partial_prefix_sums_per_radix){

	int lowestRadixForBlock = NUM_RADICES_PER_BLOCK*blockIdx.x ;
	int highestRadixForBlock = lowestRadixForBlock + NUM_RADICES_PER_BLOCK - 1;
    //Both radices are included in the radix range for this group
	__shared__ int shared_counters[NUM_RADICES_PER_BLOCK][NUM_GROUPS];
	if(threadIdx.x < NUM_RADICES_PER_BLOCK)
	{
		int i = threadIdx.x;
		for(int j = 0; j < NUM_GROUPS; j++)
		{
			shared_counters[i][j] = 0;
		}
	}
	__syncthreads();

	int last_value = 0;
	if(threadIdx.x <= highestRadixForBlock - lowestRadixForBlock)
	{
		int i = threadIdx.x + lowestRadixForBlock;
		for (int j = 0; j < NUM_GROUPS - 1; ++j)
		{
			shared_counters[i - lowestRadixForBlock][j + 1] = d_counters[getAddress(i,j/NUMBER_OF_GROUPS_PER_BLOCK, j%NUMBER_OF_GROUPS_PER_BLOCK)];
		}	
		int j = NUM_GROUPS - 1;
		last_value = d_counters[getAddress(i,j/NUMBER_OF_GROUPS_PER_BLOCK, j%NUMBER_OF_GROUPS_PER_BLOCK)];
	}
	__syncthreads();

    //Prefix sum naive implementation for shared memory
	if(threadIdx.x <= highestRadixForBlock - lowestRadixForBlock)
	{
		int i = threadIdx.x + lowestRadixForBlock;
		for(int j = 1;j<NUM_GROUPS; j++){
			shared_counters[i - lowestRadixForBlock][j] = shared_counters[i - lowestRadixForBlock][j] + shared_counters[i - lowestRadixForBlock][j-1];
		}
	}
	__syncthreads();

	if(threadIdx.x <= highestRadixForBlock - lowestRadixForBlock)
	{
		int i = threadIdx.x + lowestRadixForBlock;
		for (int j = 0; j < NUM_GROUPS; ++j)
		{
			d_counters[getAddress(i,j/NUMBER_OF_GROUPS_PER_BLOCK, j%NUMBER_OF_GROUPS_PER_BLOCK)] = shared_counters[i - lowestRadixForBlock][j];
		}
		d_partial_prefix_sums_per_radix[i] = shared_counters[i - lowestRadixForBlock][NUM_GROUPS-1] + last_value;
	}


}

void launch_kernel_phase_2(int* d_cellID, int * d_objectID, int size, int pass, int* d_counters, int* d_partial_prefix_sums_per_radix){
    //keep launch config of all kernels same, except kernel 2
	dim3 grid(NUM_RADICES/NUM_RADICES_PER_BLOCK);
	dim3 block(BLOCK_DIM_SORT,1);
	int partition_size = ceil((float)size/(grid.x*block.x));
	int Num_Elements_Per_Group = R*partition_size;
	phase_2_kernel <<<grid, block>>>(d_cellID, d_objectID, size, partition_size, Num_Elements_Per_Group, pass, d_counters, d_partial_prefix_sums_per_radix);
}


void __global__ phase_3_kernel(int *d_cellID, int *d_objectID, int size, int partition_size, int Num_Elements_Per_Group, int pass, int * d_counters, int* d_partial_prefix_sums_per_radix, int * d_sorted_cellID, int* d_sorted_objectID){

	__shared__ int shared_parallel_prefix[NUM_RADICES];
	__shared__ int shared_counters[NUMBER_OF_GROUPS_PER_BLOCK][NUM_RADICES];


	if(threadIdx.x%R==0){
		for (int i = 0; i < NUM_RADICES; ++i){
			shared_counters[threadIdx.x/R][i] = 0;
		}
	}
	__syncthreads();

	if(threadIdx.x == 0)
	{
		shared_parallel_prefix[0] = 0;
		for (int i = 0; i < NUM_RADICES - 1; ++i)
		{
			shared_parallel_prefix[i+1] = d_partial_prefix_sums_per_radix[i];
		}

	// }
	// __syncthreads();

    // if(threadIdx.x == 0){
		for (int i = 1; i < NUM_RADICES; ++i)
		{
			shared_parallel_prefix[i] = shared_parallel_prefix[i-1] + shared_parallel_prefix[i];
		}
	}
	__syncthreads();

	if(threadIdx.x%R == 0){
		for (int i = 0; i < NUM_RADICES; ++i)
		{
			shared_counters[threadIdx.x/R][i] = d_counters[getAddress(i,blockIdx.x,threadIdx.x/R)];
			if(i>0)
				shared_counters[threadIdx.x/R][i] += shared_parallel_prefix[i];
		}
	}
	__syncthreads();

	int firstCellID =(blockIdx.x*NUMBER_OF_GROUPS_PER_BLOCK + threadIdx.x/R)*Num_Elements_Per_Group + threadIdx.x%R;
	unsigned int mask = 0;
	for (int i = 0; i < L; ++i)
		mask = mask<<1 | 1;
	mask = mask << (pass*L);
	for (int i = firstCellID; i < firstCellID + R*partition_size && i<size; i+=R)
	{
		unsigned int masked_number = d_cellID[i] & (mask);
		masked_number = masked_number >> (L*pass);
		int address_to_update = atomicInc((unsigned int*)&shared_counters[threadIdx.x/R][masked_number], INT_MAX);
		d_sorted_cellID[address_to_update] = d_cellID[i];
		d_sorted_objectID[address_to_update] = d_objectID[i];
	}

}

void launch_kernel_phase_3(int* d_cellID, int * d_objectID, int size, int pass, int* d_counters, int *d_partial_prefix_sums_per_radix, int * d_sorted_cellID, int* d_sorted_objectID){
    //keep launch config of all kernels same, except kernel 2
	dim3 grid(NUM_BLOCKS);
	dim3 block(BLOCK_DIM_SORT);
	int partition_size = ceil((float)size/(grid.x*block.x));
	int Num_Elements_Per_Group = R*partition_size;
	phase_3_kernel <<<grid, block>>>(d_cellID, d_objectID, size, partition_size, Num_Elements_Per_Group, pass, d_counters, d_partial_prefix_sums_per_radix,d_sorted_cellID, d_sorted_objectID);
}

void swap_pointers(int **p1, int **p2){
	int *t;
	t = *p1;
	*p1 = *p2;
	*p2 = t;
}

void sort(int *d_cellID, int *d_objectID, int ARRAY_SIZE){

	
	int * d_counters;
	checkCudaErrors(cudaMalloc(&d_counters, NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int)));
	int* d_partial_prefix_sums_per_radix;
	checkCudaErrors(cudaMalloc(&d_partial_prefix_sums_per_radix, sizeof(int) * NUM_RADICES));
	int *d_sorted_cellID;
	checkCudaErrors(cudaMalloc(&d_sorted_cellID, ARRAY_SIZE*sizeof(int)));
	checkCudaErrors(cudaMemset(d_sorted_cellID, 0, ARRAY_SIZE*sizeof(int)));
	int *d_sorted_objectID;
	checkCudaErrors(cudaMalloc(&d_sorted_objectID, ARRAY_SIZE*sizeof(int)));
	checkCudaErrors(cudaMemset(d_sorted_objectID, 0, ARRAY_SIZE*sizeof(int)));
	for(int i = 0; i < MAX_PASSES ; i++)
	{	
		// printf("Pass %d\n", i);
		checkCudaErrors(cudaMemset(d_counters, 0, NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int)  ));
		launch_kernel_phase_1(d_cellID, d_objectID, ARRAY_SIZE, i, d_counters);

		// int *h_d_counters;
		// h_d_counters = (int *) malloc(NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int));
		// checkCudaErrors(cudaMemcpy(h_d_counters, d_counters, NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost ));
	    // for (int l = 0; l < NUM_RADICES; ++l)
	    // {
	    // 	printf("Radix: %d Values: ", l);
	    //     for(int j = 0; j<NUM_BLOCKS; j++){
	    //         for(int k = 0; k<NUMBER_OF_GROUPS_PER_BLOCK; k++){
	    //             printf("%d,", h_d_counters[getAddress(l,j,k)]);
	    //         }
	    //         printf("\t");
	    //     }
	    //     printf("\n\n");
	    // }

		launch_kernel_phase_2(d_cellID, d_objectID, ARRAY_SIZE, i, d_counters, d_partial_prefix_sums_per_radix);

		// int *h_d_partial_prefix_sums_per_radix;
		// h_d_partial_prefix_sums_per_radix = (int*) malloc(sizeof(int) * NUM_RADICES);
		// checkCudaErrors(cudaMemcpy(h_d_partial_prefix_sums_per_radix, d_partial_prefix_sums_per_radix, sizeof(int) * NUM_RADICES, cudaMemcpyDeviceToHost));
		// for (int l = 0; l < NUM_RADICES; ++l)
		// 	printf("Radix %d: %d\n", l,h_d_partial_prefix_sums_per_radix[l]);

		// int *h_d_counters;
		// h_d_counters = (int *) malloc(NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int));
		// checkCudaErrors(cudaMemcpy(h_d_counters, d_counters, NUM_RADICES * NUM_BLOCKS * NUMBER_OF_GROUPS_PER_BLOCK * sizeof(int), cudaMemcpyDeviceToHost ));
	    // for (int l = 0; l < NUM_RADICES; ++l)
	    // {
	    // 	printf("Radix: %d Values: ", l);
	    //     for(int j = 0; j<NUM_BLOCKS; j++){
	    //         for(int k = 0; k<NUMBER_OF_GROUPS_PER_BLOCK; k++){
	    //             printf("%d,", h_d_counters[getAddress(l,j,k)]);
	    //         }
	    //         printf("\t");
	    //     }
	    //     printf("\n\n");
	    // }

		launch_kernel_phase_3(d_cellID, d_objectID, ARRAY_SIZE, i, d_counters, d_partial_prefix_sums_per_radix, d_sorted_cellID, d_sorted_objectID);
		
		// if(i != MAX_PASSES - 1){
			swap_pointers(&d_sorted_cellID, &d_cellID);
			swap_pointers(&d_sorted_objectID, &d_objectID);
		// }
	}

	// int* h_d_sorted_cellID;
	// h_d_sorted_cellID = (int *)malloc( ARRAY_SIZE* sizeof(int));
	// checkCudaErrors(cudaMemcpy(h_d_sorted_cellID, d_sorted_cellID, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost));
	// printf("Sorted Array\n");
	// for (int i = 0; i < ARRAY_SIZE; ++i)
	// {
	// 	printf("%d ", h_d_sorted_cellID[i]);
	// }
	// free(h_d_sorted_cellID);
	// printf("\n");

	// int* h_d_sorted_objectID;
	// h_d_sorted_objectID = (int *)malloc( ARRAY_SIZE* sizeof(int));
	// checkCudaErrors(cudaMemcpy(h_d_sorted_objectID, d_sorted_objectID, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost));
	// printf("\n");
	// printf("Sorted Array\n");
	// for (int i = 0; i < ARRAY_SIZE; ++i)
	// {
	// 	printf("(%d, %d), ",h_d_sorted_cellID[i], h_d_sorted_objectID[i]);
	// }
	// free(h_d_sorted_objectID);
	// printf("\n");

	checkCudaErrors(cudaFree(d_sorted_cellID));
	checkCudaErrors(cudaFree(d_sorted_objectID));
	checkCudaErrors(cudaFree(d_partial_prefix_sums_per_radix));
	checkCudaErrors(cudaFree(d_counters));

}

// struct pair{
// 	int cellID, objectID;
// };

// int comparator(const void *a, const void *b ){
// 	return (*(pair *)a).cellID - (*(pair*)b).cellID;
// }


// int main(int argc, char const *argv[])
// {
// 	// cudaSetDevice(0);
// 	int ARRAY_SIZE = 8956314;
// 	if(argc>1){
// 		ARRAY_SIZE = atoi(argv[1]);
// 		printf("Changed size to %d\n", ARRAY_SIZE);
// 	}
// 	srand(time(NULL));
// 	int *cellID = (int*) malloc(ARRAY_SIZE*sizeof(int));
// 	int *objectID = (int*) malloc(ARRAY_SIZE*sizeof(int));
// 	if(cellID == NULL || objectID == NULL)
// 	{
// 		printf("Cannot allocate host memory for sorting. Aborting!\n");
// 		exit(1);
// 	}
// 	for (int i = 0; i < ARRAY_SIZE; ++i)
// 	{
// 		cellID[i] = rand();
// 		objectID[i] = rand();
// 	}

// 	int * d_cellID, *d_objectID;
// 	checkCudaErrors(cudaMalloc(&d_cellID, ARRAY_SIZE*sizeof(int)));
// 	checkCudaErrors(cudaMalloc(&d_objectID, ARRAY_SIZE*sizeof(int)));
// 	checkCudaErrors(cudaMemcpy(d_cellID, cellID, ARRAY_SIZE*sizeof(int),cudaMemcpyHostToDevice));
// 	checkCudaErrors(cudaMemcpy(d_objectID, objectID, ARRAY_SIZE*sizeof(int), cudaMemcpyHostToDevice));
// 	sort(d_cellID, d_objectID, ARRAY_SIZE);


// 	//Retrieve the arrays back to host for testing
// 	int* h_d_sorted_cellID;
// 	h_d_sorted_cellID = (int *)malloc( ARRAY_SIZE* sizeof(int));
// 	checkCudaErrors(cudaMemcpy(h_d_sorted_cellID, d_cellID, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost));
// 	int* h_d_sorted_objectID;
// 	h_d_sorted_objectID = (int *)malloc( ARRAY_SIZE* sizeof(int));
// 	checkCudaErrors(cudaMemcpy(h_d_sorted_objectID, d_objectID, ARRAY_SIZE*sizeof(int), cudaMemcpyDeviceToHost));


// 	std::clock_t starttime = std::clock();

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

// 	//Sort and test
// 	qsort(pairs, ARRAY_SIZE, sizeof(pair), comparator);

// 	double duration = (std::clock() - starttime) / (double)CLOCKS_PER_SEC;
// 	printf("Sorting took time %f on CPU\n", duration);

// 	bool flag = true;
// 	for (int i = 0; i < ARRAY_SIZE; ++i)
// 	{
// 		if(h_d_sorted_cellID[i] != pairs[i].cellID && h_d_sorted_objectID[i] != pairs[i].objectID){
// 			flag = false; break;
// 		}
// 	}
// 	if (flag) printf("AC\n");
// 	else printf("WA\n");
// 	free(h_d_sorted_cellID);
// 	free(h_d_sorted_objectID);


// 	// //Printing Code
// 	// printf("Sorted Array\n");
// 	// for (int i = 0; i < ARRAY_SIZE; ++i)
// 	// {
// 	// 	printf("%d ", h_d_sorted_cellID[i]);
// 	// }	
// 	// printf("\n");
// 	// printf("Sorted Array\n");
// 	// for (int i = 0; i < ARRAY_SIZE; ++i)
// 	// {
// 	// 	printf("(%d, %d), ",h_d_sorted_cellID[i], h_d_sorted_objectID[i]);
// 	// }
// 	// printf("\n");



// 	return 0;
// }
