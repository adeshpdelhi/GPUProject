
__global__ void createPCSAndCallNarrowPhase(int * d_cellID, int *d_object_ID, int partition_size, int total_size){
	int block_start = blockIdx.x * blockDim.x;
	int block_end = (blockIdx.x + 1) * blockDim.x;
	int thread_start = block_start + threadIdx.x;
	__shared__ shared_d_cellID[blockDim.x], shared_d_objectID[blockDim.x];
	for (int i = thread_start; i < partition_size; ++i)
	{
		if( i >= total_size )
			continue;
		shared_d_cellID[i] = d_cellID[i];
		shared_d_objectID[i] = d_objectID[i];
	}
	__syncthreads();
	if(thread_start == 0){
		/* Exceptionally make the first thread consider the very first transition as well*/
	}
	bool flag_found_first_transition = false;
	int index_first_transition = -1;
	for(int i = thread_start; i < thread_start + partition_size; i++){
		if( i >= total_size )
			continue;
		if( i == block_end - 1 ){
			if(d_cellID[i + 1] & 0x7FFFFFFF != shared_d_cellID[i] & 0x7FFFFFFF){
				/*First transition detected*/
				flag_found_first_transition = true;
				index_first_transition = i + 1;
			}
		}
		if( shared_d_cellID[i + 1] & 0x7FFFFFFF != shared_d_cellID[i] & 0x7FFFFFFF ){
			/*First transition detected*/
			flag_found_first_transition = true;
			index_first_transition = i + 1;
		}
	}
	if(flag_found_first_transition){
		int i = index_first_transition;
		bool second_transition_found = false;
		while(second_transition_found == false){
			if(i < block_end - 1){
				if( shared_d_cellID[i + 1] & 0x7FFFFFFF != shared_d_cellID[i] & 0x7FFFFFFF ){
					/*First transition detected*/
					flag_found_first_transition = true;
					index_first_transition = i + 1;
				}
			}
			else{

			}
		}
	}
}