#include "object.h"

bool loadOBJ(
    const char * path, 
    std::vector<glm::vec3> & out_vertices, std::vector<unsigned int> &temp_mappings
){
    printf("Loading OBJ file %s...\n", path);



    FILE * file = fopen(path, "r");
    if( file == NULL ){
        printf("Impossible to open the file ! Are you in the right path ? See Tutorial 1 for details\n");
        getchar();
        return false;
    }

    while( 1 ){

        char lineHeader[128];
        // read the first word of the line
        int res = fscanf(file, "%s", lineHeader);
        if (res == EOF)
            break; // EOF = End Of File. Quit the loop.

        // else : parse lineHeader
        
        if ( strcmp( lineHeader, "v" ) == 0 ){
            glm::vec3 vertex;
            fscanf(file, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z );
            out_vertices.push_back(vertex);
        }
        else if ( strcmp( lineHeader, "f" ) == 0 ){
            unsigned int vertexIndex[3];
            int matches = fscanf(file, "%d %d %d\n", &vertexIndex[0], &vertexIndex[1], &vertexIndex[2]);
            if (matches != 3){
                printf("File can't be read by our simple parser :-( Try exporting with other options\n");
                return false;
            }
            temp_mappings.push_back(vertexIndex[0]-1);
            temp_mappings.push_back(vertexIndex[1]-1);
            temp_mappings.push_back(vertexIndex[2]-1);
        }else{
            // Probably a comment, eat up the rest of the line
            char stupidBuffer[1000];
            fgets(stupidBuffer, 1000, file);
        }

    }

    return true;
}

__device__ int getObjectId(int index, Object* d_objects){
    for (int i = 0; i < OBJECT_COUNT; ++i)
    {
        if(index >= d_objects[i].start_index && index < d_objects[i].start_index + d_objects[i].n_vertices)
        {
            return i;
        }
    }
    return -1;
}


// void appendObject(std::vector<glm::vec3> &vertices, std::vector<unsigned int> &mappings,
//  std::vector<glm::vec3> &temp_vertices, std::vector<unsigned int> &temp_mappings) {

//     int startIndex = vertices.size();
//     vertices.insert(vertices.end(), temp_vertices.begin(), temp_vertices.end());
//     for (int i = 0; i < temp_mappings.size(); ++i)
//     {
//         mappings.push_back(temp_mappings[i] + startIndex);
//     }
// }

