#ifndef OBJECT_H
#define OBJECT_H

#include <bits/stdc++.h>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <ctime>
#include <climits>

#include "openGL_Cuda_headers.h"
#include "glm/glm.hpp"
#include "constants.h"
#include "helperStructures.cu"


class Template{
private:
	std::string name;
	int n_vertices;
	int template_id;
	std::vector<glm::vec3> vertices;
	std::vector<unsigned int> triangulatedFaces;

	// GLuint vbo;
	// GLuint ibo;
	boundingVolume bV;

	float maxDistanceAlongXYZ;
	float3 centroid;
	void find_max_Distance_Along_XYZ();
	void findCentroid();
	void findBoundingVolume(float maxx, float minX, float maxY, float minY, float maxZ, float minZ);
public:
	Template(std::vector<glm::vec3> v, std::vector<unsigned int> f, std::string n, int id){
		vertices.assign(v.begin(), v.end())	;
		// for(int i = 0; i < v.size();i++){
		// 	printf("loaded vertices[%d]: [%f %f %f]\n",i , v[i].x, v[i].y, v[i].z );
		// }
		// for(int i = 0; i < vertices.size();i++){
		// 	printf("stored vertices[%d]: [%f %f %f]\n",i , vertices[i].x, vertices[i].y, vertices[i].z );
		// }
		triangulatedFaces.assign(f.begin(), f.end());
		name = n;
		template_id = id;
		n_vertices = vertices.size();
		find_max_Distance_Along_XYZ();
		findCentroid();
		// printf("Constructor - Centroid: [%f %f %f]\n", centroid.x, centroid.y, centroid.z);
	}
	
	std::vector<glm::vec3> getVertices(){
		return vertices;
	}
	std::vector<unsigned int> getFaces(){
		return triangulatedFaces;
	}
	float3 getCentroid(){
		return centroid;
	}
	float get_max_Distance_Along_XYZ(){
		return maxDistanceAlongXYZ;
	}
	int getNVertices(){
		return n_vertices;
	}
	boundingVolume getBoundingVolume(float4 shift){
		boundingVolume temp;
		temp.u_f_r = make_float3(bV.u_f_r.x + shift.x, bV.u_f_r.y + shift.y, bV.u_f_r.z + shift.z);
		temp.u_f_l = make_float3(bV.u_f_l.x + shift.x, bV.u_f_l.y + shift.y, bV.u_f_l.z + shift.z);
		
		temp.u_b_r = make_float3(bV.u_b_r.x + shift.x, bV.u_b_r.y + shift.y, bV.u_b_r.z + shift.z);
		temp.u_b_l = make_float3(bV.u_b_l.x + shift.x, bV.u_b_l.y + shift.y, bV.u_b_l.z + shift.z);
		
		temp.lo_f_r = make_float3(bV.lo_f_r.x + shift.x, bV.lo_f_r.y + shift.y, bV.lo_f_r.z + shift.z);
		temp.lo_f_l = make_float3(bV.lo_f_l.x + shift.x, bV.lo_f_l.y + shift.y, bV.lo_f_l.z + shift.z);
		
		temp.lo_b_r = make_float3(bV.lo_b_r.x + shift.x, bV.lo_b_r.y + shift.y, bV.lo_b_r.z + shift.z);
		temp.lo_b_l = make_float3(bV.lo_b_l.x + shift.x, bV.lo_b_l.y + shift.y, bV.lo_b_l.z + shift.z);
		return temp;
	}
	// void setName(string n){
	// 	name = n;
	// }
	// void setVertices(std::vector<glm::vec3> v){
	// 	vertices.assign(v.begin(), v.end());
	// }
	// void setFaces(std::vector<glm::vec3> v){
	// 	triangulatedFaces.assign(v.begin(), v.end());
	// }
};

void Template::find_max_Distance_Along_XYZ(){
	float maxX = INT_MIN, minX = INT_MAX, maxY = INT_MIN, minY = INT_MAX, maxZ = INT_MIN, minZ = INT_MAX;
	for (int i = 0; i < n_vertices; i++){
		maxX = max(maxX, vertices[i].x);
		minX = min(minX, vertices[i].x);
		maxY = max(maxY, vertices[i].y);
		minY = min(minY, vertices[i].y);
		maxZ = max(maxZ, vertices[i].z);
		minZ = min(minZ, vertices[i].z);
	}
	findBoundingVolume(maxX, minX, maxY, minY, maxZ, minZ);
	maxDistanceAlongXYZ = max(maxX-minX, max(maxY-minY, maxZ - minZ));
	// printf("get_max_Distance_Along_XYZ : %f \n", maxDistanceAlongXYZ);
}

void Template::findCentroid(){
	// printf("Finding Centroid\n");
	for(int i = 0; i < n_vertices; i++){
		centroid.x += (vertices[i].x);
		centroid.y += (vertices[i].y);
		centroid.z += (vertices[i].z);	
	}
	centroid.x /= n_vertices;
	centroid.y /= n_vertices;
	centroid.z /= n_vertices;
	// printf("centroid: [%f %f %f ]\n",centroid.x,centroid.y, centroid.z );
}

void Template::findBoundingVolume(float maxX, float minX, float maxY, float minY, float maxZ, float minZ){
	//assumption r = +ve X, f = +ve Z, u = +ve Y
	//assumption l = -ve X, b = -ve Z, lo = -ve Y

	bV.u_f_r = make_float3(maxX, maxY, maxZ);
	bV.u_f_l = make_float3(minX, maxY, maxZ);
	
	bV.u_b_r = make_float3(maxX, maxY, minZ);
	bV.u_b_l = make_float3(minX, maxY, minZ);
	
	bV.lo_f_r = make_float3(maxX, minY, maxZ);
	bV.lo_f_l = make_float3(minX, minY, maxZ);
	
	bV.lo_b_r = make_float3(maxX, minY, minZ);
	bV.lo_b_l = make_float3(minX, minY, minZ);

}

bool comp(Template a, Template b){
	return a.get_max_Distance_Along_XYZ() < b.get_max_Distance_Along_XYZ();
}

class Templates{
private:
	int number_of_templates;
	std::vector<Template> templates ;
	float max_Bounding_Box;
public:
	Templates(){
		number_of_templates = 0;
		max_Bounding_Box = INT_MIN;
		// templates.assertsign(t.begin(), t.end());
		// number_of_templates = templates.size();
	}
	void insert(Template a){
		templates.push_back(a);
		number_of_templates++;
		max_Bounding_Box = max(max_Bounding_Box, a.get_max_Distance_Along_XYZ());
		// printf("max_Bounding_Box: %f\n", max_Bounding_Box);
	}	
	Template get_ith_template(int i){
		assert(i < number_of_templates);
		return templates[i];
	}
	int getMaximumBoundingBox(){
		// printf("getMaximumBoundingBox: %f , %d\n", max_Bounding_Box, (int)std::ceil(max_Bounding_Box));
		return (int)std::ceil(max_Bounding_Box);
		// return *(std::max_element(templates.begin(), templates.end(), comp)).get_max_Distance_Along_XYZ();
	}
};

float4 getRandomSpeed(){
    return make_float4(((rand()%100)-50)/MAX_SPEED,((rand()%100)-50)/MAX_SPEED,((rand()%100)-50)/MAX_SPEED,1.0f);
}
float4 getRandomInitialLocation(){
    return make_float4(rand()%MAX_SPACE,rand()%MAX_SPACE,rand()%MAX_SPACE,1.0f);
}

Templates templates;	

class Object{
private:		
    // float rotation_matrix[4][4];
public:

	int template_id;
	int start_index;
	int n_vertices;

    float4 speed;
    float4 centroid;
    boundingVolume bV ;
    float4 initial_location;

	Object(int template_index, int s){
		template_id = template_index;
		
		start_index = s;
		n_vertices = templates.get_ith_template(template_index).getNVertices();
		speed = getRandomSpeed();
		// printf("speed: %f %f %f %f \n", speed.x, speed.y, speed.z, speed.w );

		initial_location = getRandomInitialLocation();
		// printf("initial_location: %f %f %f %f \n", initial_location.x, initial_location.y, initial_location.z, initial_location.w );


		float3 c = templates.get_ith_template(template_index).getCentroid();
		// printf("centroid_orig: %f %f %f \n", c.x, c.y, c.z );

		centroid = make_float4(c.x+initial_location.x, c.y+initial_location.y, c.z+initial_location.z, 1.0f);
		// printf("centroid: %f %f %f %f \n", centroid.x, centroid.y, centroid.z, centroid.w );

		bV = templates.get_ith_template(template_index).getBoundingVolume(initial_location);
	}
	int getTemplateId(){
		return template_id;
	}
};

class Objects{
private:
	
public:
	int curIdx;
	Object *objs;
	std::vector<float4> vertices;
    std::vector <unsigned int> mappings;
    std::vector<int> Obj_IDS;
    Objects(){
    	curIdx = 0;
    	objs = (Object *)malloc(sizeof(Object)*OBJECT_COUNT);
    }
	void insert(int template_id, int obj_id);
};

void Objects::insert(int template_id, int obj_id){
	std::vector<glm::vec3> v = templates.get_ith_template(template_id).getVertices();
	std::vector<unsigned int> m = templates.get_ith_template(template_id).getFaces();
	// for(int i = 0; i < v.size();i++){
	// 	printf("loaded template vertices[%d]: [%f %f %f]\n",i , v[i].x, v[i].y, v[i].z );
	// }
	// assert(mappings.size()+ m.size() < MAX_MAPPINGS);

	int startIndex = vertices.size();
	objs[curIdx++] = Object(template_id, startIndex);
	// objs.push_back(Object(id, startIndex, startIndex+v.size()));
	for (int i = 0; i < v.size(); i++){
		vertices.push_back(make_float4(v[i].x+objs[curIdx-1].initial_location.x,
			v[i].y+objs[curIdx-1].initial_location.y,
			v[i].z+objs[curIdx-1].initial_location.z, 1.0f));
		Obj_IDS.push_back(obj_id);
	}
	// vertices.insert(vertices.end(), v.begin(), v.end() );
	
	for (int i = 0; i < m.size(); ++i)
    {
        mappings.push_back(m[i] + startIndex);
    }
}

Objects OBJECTS;
Object *D_OBJECTS;
int *D_OBJ_IDS;

// std::string objectsToLoad[NUM_TEMPLATES] = {"cube","cone","sphere"}; 
std::string objectsToLoad[NUM_TEMPLATES] = {"cube","cone"}; 



GLuint vbo;
GLuint ibo;


float CELL_SIZE;

// vbo variables
void *d_vbo_buffer = NULL;

struct cudaGraphicsResource *cuda_vbo_resource;

// float boundingBoxLength;

float4 *host_pos; //globally declared to allot more vertices than feasible in local scope
 
// std::vector<glm::vec3> vertices;
// std::vector<GLubyte> mappings;

int *D_CELLIDS;
int *D_OBJECT_IDS;

bool loadOBJ(
    const char * path, 
    std::vector<glm::vec3> & out_vertices, 
    std::vector <unsigned int> & mappings
);

// float getMaximumBoundingBox(std::vector <std::vector <glm::vec3> > );

// void appendObject(std::vector<glm::vec3> &vertices, std::vector<unsigned int> &mappings,
//  std::vector<glm::vec3> &temp_vertices, std::vector<unsigned int> &temp_mappings);


#endif