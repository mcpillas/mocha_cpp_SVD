#include <cstdint>

int16_t mytanh(int16_t x);

void matxvec(float *w, float *b,
        float *input, float *output,
        int isize, int osize);
void matxvec(int *w, int *b,
        int *input, int *output,
        int isize, int osize);
void matxvec(int16_t *w, int16_t *b,
        int16_t *input, int16_t *output,
        int isize, int osize);
void matxvec(int8_t *w, int8_t *b,
        int8_t *input, int8_t *output,
        int isize, int osize);

void relu(float* a,   int size);
void relu(int* a,     int size);
void relu(int16_t* a, int size);
void relu(int8_t* a,  int size);

void sigmoid(float* a,   int size);
void sigmoid(int* a,     int size);
void sigmoid(int16_t* a, int size);
void sigmoid(int8_t* a,  int size);

void softmax(float* a,   int size);
void softmax(int* a,     int size);
void softmax(int16_t* a, int size);
void softmax(int8_t* a,  int size);

int max(float* a, int size);
int max(int* a, int size);
int max(int16_t* a, int size);
int max(int8_t* a, int size);

void array_tanh(float * a, int size);
void array_tanh(int* a, int size);
void array_tanh(int16_t* a, int size);
void array_tanh(int8_t* a, int size);

// input size  = (n, m);
// output size = (n, (m+1)/2);
void max_pooling(float* data, int n, int m);
void max_pooling(int* data, int n, int m);
void max_pooling(int16_t* data, int n, int m);
void max_pooling(int8_t* data, int n, int m);

void print_mat(int m, int n, float* a);
void print_mat(int m, int n, int* a);
void print_mat(int m, int n, int16_t* a);
void print_mat(int m, int n, int8_t* a);

