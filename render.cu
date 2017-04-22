#include "render.h"
#include "object.cu"
#include "setup.cu"
// #include "helper.cu"
#include "broadphase.cu"
#include "gjk.cu"
#include "sort.cu"

void launch_kernel(float4 *pos, Object* objects, float time, int n_vertices, int *D_CELLIDS, int *D_OBJECT_IDS)
{
    // execute the kernel
    // gridDim = number of Blocks  
    int gridDim = ceil((float)OBJECT_COUNT/BLOCK_DIM);
    dim3 grid(gridDim,1);
    dim3 block(BLOCK_DIM,1);
    find_CellID<<< grid, block>>>(objects, D_CELLIDS, D_OBJECT_IDS, CELL_SIZE);
    
    sort(D_CELLIDS, D_OBJECT_IDS, 8*OBJECT_COUNT);

    // bool *d_result, h_result;
    // cudaMalloc((void**)&d_result, sizeof(bool));
    // gjk<<< 1 , 1>>>(pos, objects, 0, 1, d_result);
    // // printf("result gjk : %d\n", result );
    // cudaMemcpy(&h_result, d_result,sizeof(bool), cudaMemcpyDeviceToHost);
    // printf("%d\n", h_result );

    gridDim = ceil((float)n_vertices/BLOCK_DIM);
    dim3 grid1(gridDim,1);
    dim3 block1(BLOCK_DIM,1);
    run_vbo_kernel<<< grid1, block1>>>(pos, objects, time);
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    sdkStartTimer(&timer);

    // run CUDA kernel to generate vertex positions
    runCuda(&cuda_vbo_resource);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

    // glScalef(0.001,0.01,0.001);
    // render from the vbo
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_INDEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    // glDrawSomething(GL_TRIANGLES, 0, vertices.size());
    // printf("mappings size: %d\n",mappings.size());
    glDrawElements(GL_TRIANGLES, OBJECTS.mappings.size(), GL_UNSIGNED_INT, (const GLvoid *)0);
   
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_INDEX_ARRAY);


    glutSwapBuffers();

    g_fAnim += 0.01f;

    sdkStopTimer(&timer);
    computeFPS();
}


void computeFPS()
{
    frameCount++;
    fpsCount++;

    if (fpsCount == fpsLimit)
    {
        avgFPS = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
        fpsCount = 0;
        fpsLimit = (int)MAX(avgFPS, 1.f);

        sdkResetTimer(&timer);
    }

    char fps[256];
    sprintf(fps, "Cuda GL Interop (VBO): %3.1f fps (Max 100Hz)", avgFPS);
    glutSetWindowTitle(fps);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
bool runTest(int argc, char **argv, char *ref_file)
{
    // Create the CUTIL timer
    sdkCreateTimer(&timer);

    // command line mode only
    if (ref_file != NULL)
    {
        // // This will pick the best possible CUDA capable device
        // int devID = findCudaDevice(argc, (const char **)argv);
        // printf("ref_file not found\n");
        // // create VBO
        // checkCudaErrors(cudaMalloc((void **)&d_vbo_buffer, vertices.size()*sizeof(glm::vec3)));
        // // run the cuda part
        // runAutoTest(devID, argv, ref_file);
        // // check result of Cuda step
        // checkResultCuda(argc, argv, vbo);
        // cudaFree(d_vbo_buffer);
        // d_vbo_buffer = NULL;
    }
    else
    {
        printf("ref_file found\n");
        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        if (false == initGL(&argc, argv))
        {
            return false;
        }

        // use command-line specified CUDA device, otherwise use device with highest Gflops/s
        if (checkCmdLineFlag(argc, (const char **)argv, "device"))
        {
            if (gpuGLDeviceInit(argc, (const char **)argv) == -1)
            {
                return false;
            }
        }
        else
        {
            cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
        }

        // register callbacks
        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutMouseFunc(mouse);
        glutMotionFunc(motion);
#if defined (__APPLE__) || defined(MACOSX)
        atexit(cleanup);
#else
        glutCloseFunc(cleanup);
#endif

        // create VBO
        createVBOAndIBO(&vbo, &ibo, &cuda_vbo_resource, cudaGraphicsMapFlagsWriteDiscard);

        // run the cuda part
        runCuda(&cuda_vbo_resource);

        // start rendering mainloop
        glutMainLoop();
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda(struct cudaGraphicsResource **vbo_resource)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *d_ptr;
    checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
    size_t num_bytes;
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &num_bytes, *vbo_resource));
    // printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);
    launch_kernel(d_ptr, D_OBJECTS, g_fAnim, OBJECTS.vertices.size(), D_CELLIDS, D_OBJECT_IDS);    
    // unmap buffer object
    checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
}

////////////////////////////////////////////////////////////////////////////////
//! Create VBO
////////////////////////////////////////////////////////////////////////////////
void createVBOAndIBO(GLuint *vbo, GLuint *ibo, struct cudaGraphicsResource **vbo_res,
               unsigned int vbo_res_flags)
{
    assert(OBJECT_COUNT > 0);
    for(int i = 0; i < NUM_TEMPLATES; i++){
        std::vector<glm::vec3> tempvertices;
        std::vector<unsigned int> tempIndices;
        bool res = loadOBJ(std::string("./Objects/"+objectsToLoad[i]+".obj").c_str(), tempvertices, tempIndices);
        assert(res);
        templates.insert(Template(tempvertices, tempIndices,objectsToLoad[i],i));
    }
    
    CELL_SIZE = 1.5*(templates.getMaximumBoundingBox());

    for (int i = 0; i < OBJECT_COUNT; ++i)
    {
        OBJECTS.insert(i%2);        
    }
    printf("Size of vertices = %d\n", OBJECTS.vertices.size());
    printf("Size of mappings = %d\n", OBJECTS.mappings.size());
    //Code crashes at the following line for object count > 1500

    assert(vbo);
    // create buffer object
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, *vbo);
    // initialize buffer object
    glBufferData(GL_ARRAY_BUFFER, OBJECTS.vertices.size()*sizeof(float4), &OBJECTS.vertices[0], GL_DYNAMIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    printf("Object allocated in OpenGL\n");

    checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

    SDK_CHECK_ERROR_GL();

    assert(ibo);
    // create buffer object
    glGenBuffers(1, ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ibo);
    // initialize buffer object
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, OBJECTS.mappings.size()*sizeof(unsigned int), &OBJECTS.mappings[0], GL_STATIC_DRAW);
    // glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);


    checkCudaErrors(cudaMalloc(&D_OBJECTS, sizeof(Object)*OBJECT_COUNT));
    checkCudaErrors(cudaMemcpy(D_OBJECTS, OBJECTS.objs, sizeof(Object)*OBJECT_COUNT, cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc(&D_CELLIDS, sizeof(int)*8*OBJECT_COUNT));
    checkCudaErrors(cudaMalloc(&D_OBJECT_IDS, sizeof(int)*8*OBJECT_COUNT));

    // checkCudaErrors(cudaMemcpy(D_OBJECTS, OBJECTS.objs, sizeof(Object)*OBJECT_COUNT, cudaMemcpyHostToDevice));

    
    SDK_CHECK_ERROR_GL();
}


// void runCuda(struct cudaGraphicsResource **vbo_resource)
// {
//     // map OpenGL buffer object for writing from CUDA
//     float4 *d_ptr;
//     checkCudaErrors(cudaGraphicsMapResources(1, vbo_resource, 0));
//     size_t num_bytes;
//     checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void **)&d_ptr, &num_bytes, *vbo_resource));
//     // printf("CUDA mapped VBO: May access %ld bytes\n", num_bytes);

//     for (int i = 0; i<vertices.size(); ++i)
//     {
//         host_pos[i] = make_float4(vertices[i].x,vertices[i].y, vertices[i].z,1.0f);
//     }
//     checkCudaErrors(cudaMemcpy(d_ptr, host_pos, vertices.size()*sizeof(float4), cudaMemcpyHostToDevice));
    
//     struct object* d_objects;
//     checkCudaErrors(cudaMalloc(&d_objects, sizeof(objects)));
    
//     checkCudaErrors(cudaMemcpy(d_objects, objects, sizeof(objects), cudaMemcpyHostToDevice));
//     launch_kernel(d_ptr, d_objects, g_fAnim);
//     checkCudaErrors(cudaFree(d_objects));
//     // unmap buffer object
//     checkCudaErrors(cudaGraphicsUnmapResources(1, vbo_resource, 0));
// }

#ifdef _WIN32
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) fopen_s(&fHandle, filename, mode)
#endif
#else
#ifndef FOPEN
#define FOPEN(fHandle,filename,mode) (fHandle = fopen(filename, mode))
#endif
#endif

void sdkDumpBin2(void *data, unsigned int bytes, const char *filename)
{
    printf("sdkDumpBin: <%s>\n", filename);
    FILE *fp;
    FOPEN(fp, filename, "wb");
    fwrite(data, bytes, 1, fp);
    fflush(fp);
    fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runAutoTest(int devID, char **argv, char *ref_file)
{
    char *reference_file = NULL;
    void *imageData = malloc(mesh_width*mesh_height*sizeof(float));

    // execute the kernel
    launch_kernel((float4 *)d_vbo_buffer, OBJECTS.objs, g_fAnim, 10, NULL, NULL);

    cudaDeviceSynchronize();
    getLastCudaError("launch_kernel failed");

    checkCudaErrors(cudaMemcpy(imageData, d_vbo_buffer, mesh_width*mesh_height*sizeof(float), cudaMemcpyDeviceToHost));

    sdkDumpBin2(imageData, mesh_width*mesh_height*sizeof(float), "simpleGL.bin");
    reference_file = sdkFindFilePath(ref_file, argv[0]);

    if (reference_file &&
        !sdkCompareBin2BinFloat("simpleGL.bin", reference_file,
                                mesh_width*mesh_height*sizeof(float),
                                MAX_EPSILON_ERROR, THRESHOLD, pArgv[0]))
    {
        g_TotalErrors++;
    }
}

// void createVBOAndIBO(GLuint *vbo, GLuint *ibo, struct cudaGraphicsResource **vbo_res,
//                unsigned int vbo_res_flags)
// {
//     for(int i = 0; i < NUM_TEMPLATES; i++){
//         std::std::vector<glm::vec3> temp;
//         res = loadOBJ("./Objects/"+objectsToLoad[i]+".obj", temp);
//     }

//     // std::vector<glm::vec3> temp_vertices_cube;
//     // std::vector<glm::vec3> temp_vertices_cone;
    
//     // std::vector<unsigned int> temp_mappings_cube;
//     // std::vector<unsigned int> temp_mappings_cone;
//     // if(OBJECT_COUNT  > 0){
//     //     bool res;
//     //     res = loadOBJ("./Objects/cube.obj", temp_vertices_cube, temp_mappings_cube);
//     //     assert(res);
//     //     res = loadOBJ("./Objects/cone.obj", temp_vertices_cone, temp_mappings_cone);
//     //     assert(res);
//     //     std::vector <std::vector <glm::vec3> > concatenated_vectices;
//     //     concatenated_vectices.push_back(temp_vertices_cube);
//     //     concatenated_vectices.push_back(temp_vertices_cone);
//     //     boundingBoxLength = getMaximumBoundingBox(concatenated_vectices);
//     //     printf("Maximum bounding length: %f\n", boundingBoxLength);
//     // }
    
//     assert(vbo);
//     // create buffer object
//     glGenBuffers(1, vbo);
//     glBindBuffer(GL_ARRAY_BUFFER, *vbo);
//     // initialize buffer object
    
//     vertices.clear();
//     mappings.clear();
    
//     for (int i = 0; i < OBJECT_COUNT; ++i)
//     {
//         if(mappings.size()>MAX_MAPPINGS){
//             printf("Error! Mappings more than the threshold at object number %d. Exiting.\n", i);
//             exit(-1);
//         }
//         if(i%2 == 0){
//             objects[i].n_vertices = temp_vertices_cube.size();
//             objects[i].speed = getRandomSpeed();
//             appendObject(vertices, mappings, temp_vertices_cube, temp_mappings_cube);
//         }
//         else
//         {
//             objects[i].n_vertices = temp_vertices_cone.size();
//             objects[i].speed = getRandomSpeed();
//             appendObject(vertices, mappings, temp_vertices_cone, temp_mappings_cone);
//         }
        
//     }
//     printf("Size of vertices = %d\n", vertices.size());
//     printf("Size of mappings = %d\n", mappings.size());
//     //Code crashes at the following line for object count > 1500
//     glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float4), &vertices[0], GL_DYNAMIC_DRAW);
//     // glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_DYNAMIC_DRAW);
//     glBindBuffer(GL_ARRAY_BUFFER, 0);
//     printf("Object allocated in OpenGL\n");
//     // register this buffer object with CUDA
//     checkCudaErrors(cudaGraphicsGLRegisterBuffer(vbo_res, *vbo, vbo_res_flags));

//     SDK_CHECK_ERROR_GL();

//     assert(ibo);
//     // create buffer object
//     glGenBuffers(1, ibo);
//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *ibo);
//     // initialize buffer object
    
//     glBufferData(GL_ELEMENT_ARRAY_BUFFER, mappings.size() * sizeof(unsigned int), &mappings[0], GL_STATIC_DRAW);

//     // glBindBuffer(GL_ARRAY_BUFFER, 0);
//     glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

//     SDK_CHECK_ERROR_GL();

// }

////////////////////////////////////////////////////////////////////////////////
//! Delete VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVBOAndIBO(GLuint *vbo, GLuint *ibo, struct cudaGraphicsResource *vbo_res)
{

    // unregister this buffer object with CUDA
    checkCudaErrors(cudaGraphicsUnregisterResource(vbo_res));

    glBindBuffer(1, *vbo);
    glDeleteBuffers(1, vbo);

    glBindBuffer(1, *ibo);
    glDeleteBuffers(1, ibo);

    checkCudaErrors(cudaFree(D_OBJECTS));
    checkCudaErrors(cudaFree(D_CELLIDS));
    checkCudaErrors(cudaFree(D_OBJECT_IDS));

    *vbo = 0;
    *ibo = 0;
}

void cleanup()
{
    sdkDeleteTimer(&timer);

    if (vbo)
    {
        deleteVBOAndIBO(&vbo, &ibo, cuda_vbo_resource);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Check if the result is correct or write data to file for external
//! regression testing
////////////////////////////////////////////////////////////////////////////////
void checkResultCuda(int argc, char **argv, const GLuint &vbo)
{
    if (!d_vbo_buffer)
    {
        checkCudaErrors(cudaGraphicsUnregisterResource(cuda_vbo_resource));

        // map buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        float *data = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);

        // check result
        if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
        {
            // write file for regression test
            sdkWriteFile<float>("./data/regression.dat",
                                data, mesh_width * mesh_height * 3, 0.0, false);
        }

        // unmap GL buffer object
        if (!glUnmapBuffer(GL_ARRAY_BUFFER))
        {
            fprintf(stderr, "Unmap buffer failed.\n");
            fflush(stderr);
        }

        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo,
                                                     cudaGraphicsMapFlagsWriteDiscard));

        SDK_CHECK_ERROR_GL();
    }
}



/*
//Useless Functions -  functions not being used.
bool checkHW(char *name, const char *gpuType, int dev)
{
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    strcpy(name, deviceProp.name);

    if (!STRNCASECMP(deviceProp.name, gpuType, strlen(gpuType)))
    {
        return true;
    }
    else
    {
        return false;
    }
}

int findGraphicsGPU(char *name)
{
    int nGraphicsGPU = 0;
    int deviceCount = 0;
    bool bFoundGraphics = false;
    char firstGraphicsName[256], temp[256];

    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("> FAILED %s sample finished, exiting...\n", sSDKsample);
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("> There are no device(s) supporting CUDA\n");
        return false;
    }
    else
    {
        printf("> Found %d CUDA Capable Device(s)\n", deviceCount);
    }

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        bool bGraphics = !checkHW(temp, (const char *)"Tesla", dev);
        printf("> %s\t\tGPU %d: %s\n", (bGraphics ? "Graphics" : "Compute"), dev, temp);

        if (bGraphics)
        {
            if (!bFoundGraphics)
            {
                strcpy(firstGraphicsName, temp);
            }

            nGraphicsGPU++;
        }
    }

    if (nGraphicsGPU)
    {
        strcpy(name, firstGraphicsName);
    }
    else
    {
        strcpy(name, "this hardware");
    }

    return nGraphicsGPU;
}

*/