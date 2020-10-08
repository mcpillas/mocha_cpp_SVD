#include <cstdint>

//#define RISCV 1
#define NEON  1

void rearrange_weight(float* w, float* w_, int d_in, int d_out);
void rearrange_weight(int32_t* w, int32_t* w_, int d_in, int d_out);
void rearrange_weight(int16_t* w, int16_t* w_, int d_in, int d_out);
void rearrange_weight(int8_t* w, int8_t* w_, int d_in, int d_out);


void matxvec_v(float *w, float *b,
		float *x, float *y,
		int d_x, int d_y);

void matxvec_v(int *w, int *b,
		int *x, int *y,
		int d_x, int d_y);

void matxvec_v(int16_t *w, int16_t *b,
		int16_t *x, int16_t *y,
		int d_x, int d_y);

void matxvec_v(int8_t *w, int8_t *b,
		int8_t *x, int8_t *y,
		int d_x, int d_y);
