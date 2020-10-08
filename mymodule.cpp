#include <math.h>
#include "mymodule.h"
#include <cstdio>

int16_t tanh_lut[888] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,66,67,68,69,70,71,72,73,74,75,76,77,77,78,79,80,81,82,83,84,85,86,86,87,88,89,90,91,92,93,93,94,95,96,97,98,99,99,100,101,102,103,104,105,105,106,107,108,109,109,110,111,112,113,114,114,115,116,117,118,118,119,120,121,121,122,123,124,125,125,126,127,128,128,129,130,131,131,132,133,133,134,135,136,136,137,138,138,139,140,141,141,142,143,143,144,145,145,146,147,147,148,149,149,150,151,151,152,153,153,154,155,155,156,156,157,158,158,159,160,160,161,161,162,163,163,164,164,165,166,166,167,167,168,168,169,170,170,171,171,172,172,173,173,174,174,175,176,176,177,177,178,178,179,179,180,180,181,181,182,182,183,183,184,184,185,185,186,186,187,187,187,188,188,189,189,190,190,191,191,192,192,192,193,193,194,194,195,195,195,196,196,197,197,197,198,198,199,199,199,200,200,201,201,201,202,202,203,203,203,204,204,204,205,205,205,206,206,206,207,207,208,208,208,209,209,209,210,210,210,211,211,211,211,212,212,212,213,213,213,214,214,214,215,215,215,215,216,216,216,217,217,217,217,218,218,218,219,219,219,219,220,220,220,220,221,221,221,221,222,222,222,222,223,223,223,223,224,224,224,224,225,225,225,225,225,226,226,226,226,227,227,227,227,227,228,228,228,228,228,229,229,229,229,229,230,230,230,230,230,231,231,231,231,231,232,232,232,232,232,232,233,233,233,233,233,233,234,234,234,234,234,234,235,235,235,235,235,235,236,236,236,236,236,236,236,237,237,237,237,237,237,237,238,238,238,238,238,238,238,239,239,239,239,239,239,239,239,240,240,240,240,240,240,240,240,241,241,241,241,241,241,241,241,241,242,242,242,242,242,242,242,242,242,243,243,243,243,243,243,243,243,243,243,243,244,244,244,244,244,244,244,244,244,244,245,245,245,245,245,245,245,245,245,245,245,245,246,246,246,246,246,246,246,246,246,246,246,246,246,247,247,247,247,247,247,247,247,247,247,247,247,247,247,247,248,248,248,248,248,248,248,248,248,248,248,248,248,248,248,248,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,251,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,252,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,254,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};

int16_t mytanh(int16_t x){

	int sign, abs;
	sign = x < 0 ? -1 : 1;
	abs = sign*x;

	return sign*(abs > 887 ? 256 : tanh_lut[abs]);

}
void matxvec(float *w, float *b,
		float *input, float *output,
		int isize, int osize){

	int j, k;

	for(j=0; j<osize; j++){
		output[j] = (b==0) ? 0 : b[j];
		for(k=0; k<isize; k++){
			output[j] += w[j*isize+k]*input[k];
		}
	}
}



void matxvec(int *w, int *b,
		int *input, int *output,
		int isize, int osize) {

	int j, k;

	for(j=0; j<osize; j++){
		output[j] = (b==0) ? 0 : b[j];
		for(k=0; k<isize; k++){
			output[j] += w[j*isize+k]*input[k];
		}
	}
}


void matxvec(int16_t *w, int16_t *b,
		int16_t *input, int16_t *output,
		int isize, int osize) {

	int j, k;

	for(j=0; j<osize; j++){
		output[j] = (b==0) ? 0 : b[j];
		for(k=0; k<isize; k++){
			output[j] += w[j*isize+k]*input[k];
		}
	}
}

void matxvec(int8_t *w, int8_t *b,
		int8_t *input, int8_t *output,
		int isize, int osize) {

	int j, k;

	for(j=0; j<osize; j++){
		output[j] = (b==0) ? 0 : b[j];
		for(k=0; k<isize; k++){
			output[j] += w[j*isize+k]*input[k];
		}
	}
}

void relu(float * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = a[i] > 0 ? a[i] : 0;
	}
}

void relu(int * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = a[i] > 0 ? a[i] : 0;
	}
}

void relu(int16_t * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = a[i] > 0 ? a[i] : 0;
	}
}

void relu(int8_t * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = a[i] > 0 ? a[i] : 0;
	}
}

void sigmoid(float * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = 1.0 / (1.0 + expf(-a[i]));
	}
}

void sigmoid(int * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = 1.0 / (1.0 + expf(-a[i]));
	}
}

void sigmoid(int16_t * a, int size){

	int i;
	for(i=0; i<size; i++){
//		a[i] = 1.0 / (1.0 + expf(-a[i]));
		a[i] = (mytanh(a[i]>>1) + 256) >> 1;
	}
}

void sigmoid(int8_t * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = 1.0 / (1.0 + expf(-a[i]));
	}
}

int max(float * a, int size){

	int i;
	float max_val = -10e9;
	int max_idx;

	for(i=0; i<size; i++){
		if(a[i] > max_val){
			max_val = a[i];
			max_idx = i;
		}
	}

	return max_idx;
}

int max(int * a, int size){

	int i;
	int max_val = -1000000;
	int max_idx;

	for(i=0; i<size; i++){
		if(a[i] > max_val){
			max_val = a[i];
			max_idx = i;
		}
	}

	return max_idx;
}

int max(int16_t * a, int size){

	int i;
	int16_t max_val = -10000;
	int max_idx;

	for(i=0; i<size; i++){
		if(a[i] > max_val){
			max_val = a[i];
			max_idx = i;
		}
	}

	return max_idx;
}


int max(int8_t * a, int size){

	int i;
	int8_t max_val = -128;
	int max_idx;

	for(i=0; i<size; i++){
		if(a[i] > max_val){
			max_val = a[i];
			max_idx = i;
		}
	}

	return max_idx;
}

void array_tanh(float * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = tanhf(a[i]);
	}
}

void array_tanh(int * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = (int)tanhf((float)a[i]);
	}
}

void array_tanh(int16_t * a, int size){

	int i;
	for(i=0; i<size; i++){
		//a[i] = (int16_t)tanhf((float)a[i]);
		a[i] = mytanh(a[i]);
	}
}

void array_tanh(int8_t * a, int size){

	int i;
	for(i=0; i<size; i++){
		a[i] = (int8_t)tanhf((float)a[i]);
	}
}

// column-major
void print_mat(int m, int n, float* a){

	int i,j;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			printf("%12.4e", a[i+j*m]);
			if(n>12 && j==5){
				printf(" ... ");
				j=n-5;
			}
		}
		printf("\n");
		if(m>12 && i==5){
			printf("    ...\n");
			i=m-5;
		}
	}
}

void print_mat(int m, int n, int* a){

	int i,j;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			printf("%12d", a[i+j*m]);
			if(n>12 && j==5){
				printf(" ... ");
				j=n-5;
			}
		}
		printf("\n");
		if(m>12 && i==5){
			printf("    ...\n");
			i=m-5;
		}
	}
}


void print_mat(int m, int n, int16_t* a){

	int i,j;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			printf("%12d", a[i+j*m]);
			if(n>12 && j==5){
				printf(" ... ");
				j=n-5;
			}
		}
		printf("\n");
		if(m>12 && i==5){
			printf("    ...\n");
			i=m-5;
		}
	}
}


void print_mat(int m, int n, int8_t* a){

	int i,j;

	for(i=0; i<m; i++){
		for(j=0; j<n; j++){
			printf("%12d", a[i+j*m]);
			if(n>12 && j==5){
				printf(" ... ");
				j=n-5;
			}
		}
		printf("\n");
		if(m>12 && i==5){
			printf("    ...\n");
			i=m-5;
		}
	}
}

void max_pooling(float* data, int n, int m){

	int i, j;

	for(i=0; i<m-1; i+=2){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j] > data[n*(i+1)+j] ?
							data[n*i+j] : data[n*(i+1)+j];
		}
	}
	if(m>i){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j];
		}
	}

	return;
}
void max_pooling(int* data, int n, int m){

	int i, j;

	for(i=0; i<m-1; i+=2){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j] > data[n*(i+1)+j] ?
							data[n*i+j] : data[n*(i+1)+j];
		}
	}
	if(m>i){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j];
		}
	}

	return;
}
void max_pooling(int16_t* data, int n, int m){

	int i, j;

	for(i=0; i<m-1; i+=2){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j] > data[n*(i+1)+j] ?
							data[n*i+j] : data[n*(i+1)+j];
		}
	}
	if(m>i){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j];
		}
	}

	return;
}
void max_pooling(int8_t* data, int n, int m){

	int i, j;

	for(i=0; i<m-1; i+=2){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j] > data[n*(i+1)+j] ?
							data[n*i+j] : data[n*(i+1)+j];
		}
	}
	if(m>i){
		for(j=0; j<n; j++){
			data[n*i/2+j] = data[n*i+j];
		}
	}

	return;
}



void softmax(float* a,   int size){

	float mean = 0;
	float norm = 0;

	for(int i=0; i<size; i++){
		mean += a[i];
	}
	mean /= size;
	for(int i=0; i<size; i++){
		a[i] -= mean;
	}

	for(int i=0; i<size; i++){
		norm += expf(a[i]);
	}

	for(int i=0; i<size; i++){
		a[i] = expf(a[i]) / norm;
	}

	return;
	
	
}

