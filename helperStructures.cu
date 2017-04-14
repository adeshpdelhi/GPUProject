struct boundingVolume{
	float3 u_f_r;
	float3 u_f_l;
	float3 u_b_r;
	float3 u_b_l;
	float3 lo_f_r;
	float3 lo_f_l;
	float3 lo_b_r;
	float3 lo_b_l;
};
// class CellIDs{
// public:

// 	int *cellIDArray;
// 	int *objectIDArray;

// 	CellIDs(){
// 		cellIDArray = (int*)malloc(sizeof(int)*8*OBJECT_COUNT);
// 		objectIDArray = (int*)malloc(sizeof(int)*8*OBJECT_COUNT);
// 	}
// };

// CellIDs h_cellIDs, d_cellIDS;
