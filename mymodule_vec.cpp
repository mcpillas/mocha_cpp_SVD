#include "mymodule_vec.h"
#include <iostream>
#include "mymodule.h"

void rearrange_weight(float* w, float* w_, int d_in, int d_out){

    int i, j, k;

	int q, r;
	q = d_out / 4;
	r = d_out % 4;


    for(i=0; i<q; i++){
        for(j=0; j<d_in; j++){
            for(k=0; k<4; k++){
                w_[i*4*d_in + j*4+k] = w[4*d_in*i + k*d_in+j];
            }
        }
    }

	w  = w  + d_in*q*4;
	w_ = w_ + d_in*q*4;
    for(i=0; i<r*d_in; i++){
		w_[i] = w[i];
    }


}

void rearrange_weight(int32_t* w, int32_t* w_, int d_in, int d_out){

    int i, j, k;

    for(i=0; i<d_out/8; i++){
        for(j=0; j<d_in; j++){
            for(k=0; k<8; k++){
                w_[i*8*d_in + j*8+k] = w[8*d_in*i + k*d_in+j];
            }
        }
    }

}

void rearrange_weight(int16_t* w, int16_t* w_, int d_in, int d_out){

    int i, j, k;

	int q, r;
	q = d_out / 8;
	r = d_out % 8;


    for(i=0; i<q; i++){
        for(j=0; j<d_in; j++){
            for(k=0; k<8; k++){
                w_[i*8*d_in + j*8+k] = w[8*d_in*i + k*d_in+j];
            }
        }
    }

	w  = w  + d_in*q*8;
	w_ = w_ + d_in*q*8;
    for(i=0; i<r*d_in; i++){
		w_[i] = w[i];
    }

}

void rearrange_weight(int8_t* w, int8_t* w_, int d_in, int d_out){

    int i, j, k;

    for(i=0; i<d_out/16; i++){
        for(j=0; j<d_in; j++){
            for(k=0; k<16; k++){
                w_[i*16*d_in + j*16+k] = w[16*d_in*i + k*d_in+j];
            }
        }
    }

}


// y = w*x + b
void matxvec_v(float *w, float *b,
		float *x, float *y,
		int d_x, int d_y){

	if( d_x<1 || d_y<1 ){
		return;
	}
	

#ifdef NEON
	int i, j, k;
	float *w_ptr, *b_ptr, *o_ptr;

	int out_q, out_r;
	int in_q, in_r;

	out_q = d_y / 16 * 16;
	out_r = d_y % 16;
	in_q  = d_x / 4 * 4;
	in_r  = d_x % 4;

	for(j=0; j<out_q; j+=16){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"mov       r4,   %[x]          \n\t"
			"mov       r0,   %[w]          \n\t"
			"mov       r5,   %[b]          \n\t"
			"add       r1,   r0,   %[ws]   \n\t"
			"add       r2,   r1,   %[ws]   \n\t"
			"add       r3,   r2,   %[ws]   \n\t"
			"cmp       r5,   #0            \n\t"
			"beq       .matxvecvf32_1_L0   \n\t" // load biases to acc registers
			"vld1.32   {q8}, [r5]!         \n\t"
			"vld1.32   {q1}, [r5]!         \n\t"
			"vld1.32   {q2}, [r5]!         \n\t"
			"vld1.32   {q3}, [r5]!         \n\t"
			"b         .matxvecvf32_1_L1   \n\t"
		".matxvecvf32_1_L0:                \n\t" // set acc registers to 0
			"vmov.u32  q8,   #0            \n\t"
			"vmov.u32  q1,   #0            \n\t"
			"vmov.u32  q2,   #0            \n\t"
			"vmov.u32  q3,   #0            \n\t"
		".matxvecvf32_1_L1:                \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
		".matxvecvf32_1_L2:                \n\t"
			"vld1.32   {q0}, [r4]!         \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d0[0]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d0[0]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d0[0]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d0[0]   \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d0[1]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d0[1]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d0[1]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d0[1]   \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d1[0]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d1[0]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d1[0]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d1[0]   \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d1[1]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d1[1]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d1[1]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d1[1]   \n\t"
			"cmp       r4,   %[xend]       \n\t"
			"blt       .matxvecvf32_1_L2   \n\t"
			"cmp       %[xr],#0            \n\t"
			"beq       .matxvecvf32_1_L3   \n\t"
			"vld1.32   {q0}, [r4]!         \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d0[0]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d0[0]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d0[0]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d0[0]   \n\t"
			"cmp       %[xr],#1            \n\t"
			"beq       .matxvecvf32_1_L3   \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d0[1]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q1,   q5,   d0[1]   \n\t"
			"vld1.32   {q5}, [r1]!         \n\t"
			"vmla.f32  q2,   q6,   d0[1]   \n\t"
			"vld1.32   {q6}, [r2]!         \n\t"
			"vmla.f32  q3,   q7,   d0[1]   \n\t"
			"cmp       %[xr],#2            \n\t"
			"beq       .matxvecvf32_1_L3   \n\t"
			"vld1.32   {q7}, [r3]!         \n\t"
			"vmla.f32  q8,   q4,   d1[0]   \n\t"
			"vmla.f32  q1,   q5,   d1[0]   \n\t"
			"vmla.f32  q2,   q6,   d1[0]   \n\t"
			"vmla.f32  q3,   q7,   d1[0]   \n\t"
		".matxvecvf32_1_L3:                \n\t"
			"mov       r5,   %[y]          \n\t"
			"vst1.32   {q8}, [r5]!         \n\t"
			"vst1.32   {q1}, [r5]!         \n\t"
			"vst1.32   {q2}, [r5]!         \n\t"
			"vst1.32   {q3}, [r5]!         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[ws]"r"(d_x*16), // weight stride
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"r0","r1","r2","r3","r4","r5",
			"q0","q1","q2","q3","q4","q5","q6","q7",
			"q8"
			);
	}
	
	out_q = d_y / 4 * 4;
	for(; j<out_q; j+=4){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"mov       r4,   %[x]          \n\t"
			"mov       r0,   %[w]          \n\t"
			"mov       r5,   %[b]          \n\t"
			"cmp       r5,   #0            \n\t"
			"beq       .matxvecvf32_2_L0   \n\t" // load biases to acc registers
			"vld1.32   {q8}, [r5]!         \n\t"
			"b         .matxvecvf32_2_L2   \n\t"
		".matxvecvf32_2_L0:                \n\t" // set acc registers to 0
			"vmov.u32  q8,   #0            \n\t"
		".matxvecvf32_2_L2:                \n\t"
			"vld1.32   {q0}, [r4]!         \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d0[0]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d0[1]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d1[0]   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d1[1]   \n\t"
			"cmp       r4,   %[xend]       \n\t"
			"blt       .matxvecvf32_2_L2   \n\t"
			"cmp       %[xr],#0            \n\t"
			"beq       .matxvecvf32_2_L3   \n\t"
			"vld1.32   {q0}, [r4]!         \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d0[0]   \n\t"
			"cmp       %[xr],#1            \n\t"
			"beq       .matxvecvf32_2_L3   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d0[1]   \n\t"
			"cmp       %[xr],#2            \n\t"
			"beq       .matxvecvf32_2_L3   \n\t"
			"vld1.32   {q4}, [r0]!         \n\t"
			"vmla.f32  q8,   q4,   d1[0]   \n\t"
		".matxvecvf32_2_L3:                \n\t"
			"mov       r5,   %[y]          \n\t"
			"vst1.32   {q8}, [r5]!         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"r0","r1","r2","r3","r4","r5",
			"q0","q1","q2","q3","q4","q5","q6","q7",
			"q8"
			);

	}
	
	out_r = d_y % 4;
	if(out_r==0) return;

	b_ptr = (b==0) ? 0 : (b+j);
	w_ptr = w+(j*d_x);
	o_ptr = y+j;
	matxvec(w_ptr, b_ptr, x, o_ptr, d_x, out_r);

#elif RISCV
	int i, j, k;
	float *w_ptr, *b_ptr, *o_ptr;

	int out_q, out_r;
	int in_q, in_r;

	out_q = d_y / 16 * 16;
	out_r = d_y % 16;
	in_q  = d_x / 4 * 4;
	in_r  = d_x % 4;

	for(j=0; j<out_q; j+=16){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"addi       a3,   zero, 999    \t\n"
			"vsetvli    zero, a3,   e32    \t\n"
			"mv         a4,   %[x]         \n\t"
			"mv         a0,   %[w]         \n\t"
			"mv         a5,   %[b]         \n\t"
			"add        a1,   a0,   %[ws]  \n\t"
			"add        a2,   a1,   %[ws]  \n\t"
			"add        a3,   a2,   %[ws]  \n\t"

			"beq        a5,   zero, .matxvecvf32_1_L0\n\t"

			"vlw.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v1,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v2,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v3,   (a5)         \n\t"

			"j          .matxvecvf32_1_L1  \n\t"

		".matxvecvf32_1_L0:                \n\t"
			"vxor.vv    v8,   v8,   v8     \n\t"
			"vxor.vv    v1,   v1,   v1     \n\t"
			"vxor.vv    v2,   v2,   v2     \n\t"
			"vxor.vv    v3,   v3,   v3     \n\t"
		".matxvecvf32_1_L1:                \n\t"
			"flw        fa0,  0(a4)        \n\t"
			"flw        fa1,  4(a4)        \n\t"
			"flw        fa2,  8(a4)        \n\t"

			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"

		".matxvecvf32_1_L2:                \n\t"
			"flw        fa3,  12(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa0,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa0,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa0,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa0,  v7     \n\t"
			"flw        fa0,  16(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa1,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa1,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa1,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa1,  v7     \n\t"
			"flw        fa1,  20(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa2,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa2,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa2,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa2,  v7     \n\t"
			"flw        fa2,  24(a4)       \n\t"
			"addi       a4,   a4,   16     \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa3,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa3,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa3,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa3,  v7     \n\t"
			
			"blt        a4,   %[xend], .matxvecvf32_1_L2\n\t"

			"mv         a5,   %[xr]        \n\t"
			"beq        a5,   zero, .matxvecvf32_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa0,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa0,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa0,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa0,  v7     \n\t"
			

			"beq        a5,   zero, .matxvecvf32_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vfmacc.vf  v8,   fa1,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vfmacc.vf  v1,   fa1,  v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vfmacc.vf  v2,   fa1,  v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vfmacc.vf  v3,   fa1,  v7     \n\t"

			"beq        a5,   zero, .matxvecvf32_1_L3\n\t"
			"vfmacc.vf  v8,   fa2,  v4     \n\t"
			"vfmacc.vf  v1,   fa2,  v5     \n\t"
			"vfmacc.vf  v2,   fa2,  v6     \n\t"
			"vfmacc.vf  v3,   fa2,  v7     \n\t"
			
		".matxvecvf32_1_L3:                \n\t"
			"mv         a5,   %[y]         \n\t"
			"vsw.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v1,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v2,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v3,   (a5)         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[ws]"r"(d_x*16), // weight stride
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"a0","a1","a2","a3","a4","a5"
			,"fa0","fa1","fa2","fa3"
			);

	}
	
	out_q = d_y / 4 * 4;
	for(; j<out_q; j+=4){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"addi       a3,   zero, 999    \t\n"
			"vsetvli    zero, a3,   e32    \t\n"
			"mv         a4,   %[x]         \n\t"
			"mv         a0,   %[w]         \n\t"
			"mv         a5,   %[b]         \n\t"
			"beq        a5,   zero, .matxvecvf32_2_L0\n\t"
			"vlw.v      v8,   (a5)         \n\t"
			"j          .matxvecvf32_2_L1  \n\t"
		".matxvecvf32_2_L0:                \n\t"
			"vxor.vv    v8,   v8,   v8     \n\t"
		".matxvecvf32_2_L1:                \n\t"
			"flw        fa0,  0(a4)        \n\t"
			"flw        fa1,  4(a4)        \n\t"
			"flw        fa2,  8(a4)        \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"

		".matxvecvf32_2_L2:                \n\t"
			"flw        fa3,  12(a4)       \n\t"

			"vfmacc.vf  v8,   fa0,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"flw        fa0,  16(a4)       \n\t"

			"vfmacc.vf  v8,   fa1,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"flw        fa1,  20(a4)       \n\t"

			"vfmacc.vf  v8,   fa2,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"flw        fa2,  24(a4)       \n\t"

			"vfmacc.vf  v8,   fa3,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			
			"addi       a4,   a4,   16     \n\t"
			"blt        a4,   %[xend], .matxvecvf32_2_L2\n\t"

			"mv         a5,   %[xr]        \n\t"
			"beq        a5,   zero, .matxvecvf32_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vfmacc.vf  v8,   fa0,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"

			"beq        a5,   zero, .matxvecvf32_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vfmacc.vf  v8,   fa1,  v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"

			"beq        a5,   zero, .matxvecvf32_2_L3\n\t"
			"vfmacc.vf  v8,   fa2,  v4     \n\t"
			
		".matxvecvf32_2_L3:                \n\t"
			"mv         a5,   %[y]         \n\t"
			"vsw.v      v8,   (a5)         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"a0","a1","a2","a3","a4","a5"
			,"fa0","fa1","fa2","fa3"
			);

	}
	
	out_r = d_y % 4;
	if(out_r==0) return;

	b_ptr = (b==0) ? 0 : (b+j);
	w_ptr = w+(j*d_x);
	o_ptr = y+j;
	matxvec(w_ptr, b_ptr, x, o_ptr, d_x, out_r);
#endif

	return;
}

// y = w*x + b
void matxvec_v(int *w, int *b,
		int *x, int *y,
		int d_x, int d_y){

	if( d_x<1 || d_y<1 ){
		return;
	}


#ifdef NEON
	int i, j, k;
	for(j=0; j<d_y; j+=16){

		asm(
			"mov r4,%[i_addr]\n\t"
			"mov r0,%[w_addr]\n\t"
			"mov r5,%[b_addr]\n\t"
			"add r1, r0, %[i_size]\n\t"
			"add r2, r1, %[i_size]\n\t"
			"add r3, r2, %[i_size]\n\t"
			"vld1.32 {q8}, [r5]!\n\t"
			"vld1.32 {q1}, [r5]!\n\t"
			"vld1.32 {q2}, [r5]!\n\t"
			"vld1.32 {q3}, [r5]!\n\t"
			"vld1.32 {q4}, [r0]!\n\t"
			"vld1.32 {q5}, [r1]!\n\t"
			"vld1.32 {q6}, [r2]!\n\t"
		".matxvecvi32:\n\t"
			"vld1.32 {q0}, [r4]!\n\t"
			"vld1.32 {q7}, [r3]!\n\t"
			"vmla.i32  q8, q4, d0[0]\n\t"
			"vld1.32 {q4}, [r0]!\n\t"
			"vmla.i32  q1, q5, d0[0]\n\t"
			"vld1.32 {q5}, [r1]!\n\t"
			"vmla.i32  q2, q6, d0[0]\n\t"
			"vld1.32 {q6}, [r2]!\n\t"
			"vmla.i32  q3, q7, d0[0]\n\t"
			"vld1.32 {q7}, [r3]!\n\t"
			"vmla.i32  q8, q4, d0[1]\n\t"
			"vld1.32 {q4}, [r0]!\n\t"
			"vmla.i32  q1, q5, d0[1]\n\t"
			"vld1.32 {q5}, [r1]!\n\t"
			"vmla.i32  q2, q6, d0[1]\n\t"
			"vld1.32 {q6}, [r2]!\n\t"
			"vmla.i32  q3, q7, d0[1]\n\t"
			"vld1.32 {q7}, [r3]!\n\t"
			"vmla.i32  q8, q4, d1[0]\n\t"
			"vld1.32 {q4}, [r0]!\n\t"
			"vmla.i32  q1, q5, d1[0]\n\t"
			"vld1.32 {q5}, [r1]!\n\t"
			"vmla.i32  q2, q6, d1[0]\n\t"
			"vld1.32 {q6}, [r2]!\n\t"
			"vmla.i32  q3, q7, d1[0]\n\t"
			"vld1.32 {q7}, [r3]!\n\t"
			"vmla.i32  q8, q4, d1[1]\n\t"
			"vld1.32 {q4}, [r0]!\n\t"
			"vmla.i32  q1, q5, d1[1]\n\t"
			"vld1.32 {q5}, [r1]!\n\t"
			"vmla.i32  q2, q6, d1[1]\n\t"
			"vld1.32 {q6}, [r2]!\n\t"
			"vmla.i32  q3, q7, d1[1]\n\t"

			"cmp r4, %[end]\n\t"
			"bne .matxvecvi32\n\t"
			"mov r5,%[o_addr]\n\t"
			"vst1.32 {q8},[r5]!\n\t"
			"vst1.32 {q1},[r5]!\n\t"
			"vst1.32 {q2},[r5]!\n\t"
			"vst1.32 {q3},[r5]!\n\t"
			::
			[end]"r"(x+d_x),
			[i_size]"r"(d_x*16),
			[i_addr]"r"(x),
			[w_addr]"r"(w+(j*d_x)),
			[b_addr]"r"(b+j),
			[o_addr]"r"(y+j):
			"r0","r1","r2","r3","r4","r5",
			"q0","q1","q2","q3","q4","q5","q6","q7",
			"q8"
			//,"q9","q10","q11","q12","q13","q14","q15"
			);

	}

#elif RISCV
//// this RISCV code is not optimized
	int i, j, k;
	for(j=0; j<d_y; j+=16){

		asm(
			"mv         a4,   %[i_addr]    \n\t"
			"mv         a0,   %[w_addr]    \n\t"
			"mv         a5,   %[b_addr]    \n\t"
			"addi       a3,   zero, 4      \t\n"
			"vsetvli    zero, a3,   e32    \t\n"
			"add        a1,   a0,   %[i_size]\n\t"
			"add        a2,   a1,   %[i_size]\n\t"
			"add        a3,   a2,   %[i_size]\n\t"
			"vlw.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v1,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v2,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlw.v      v3,   (a5)         \n\t"
			"lw         t0,   0(a4)        \n\t"
			"lw         t1,   4(a4)        \n\t"
			"lw         t2,   8(a4)        \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
		".matxvecvi32:                     \n\t"
			"lw         t3,   12(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v1,   t0,   v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vmacc.vx   v2,   t0,   v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vmacc.vx   v3,   t0,   v7     \n\t"
			"lw         t0,   16(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vmacc.vx   v8,   t1,   v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v1,   t1,   v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vmacc.vx   v2,   t1,   v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vmacc.vx   v3,   t1,   v7     \n\t"
			"lw         t1,   20(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v1,   t2,   v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vmacc.vx   v2,   t2,   v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vmacc.vx   v3,   t2,   v7     \n\t"
			"lw         t2,   24(a4)       \n\t"
			"vlw.v      v7,   (a3)         \n\t"
			"addi       a3,   a3,   16     \n\t"
			"vmacc.vx   v8,   t3,   v4     \n\t"
			"vlw.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v1,   t3,   v5     \n\t"
			"vlw.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vmacc.vx   v2,   t3,   v6     \n\t"
			"vlw.v      v6,   (a2)         \n\t"
			"addi       a2,   a2,   16     \n\t"
			"vmacc.vx   v3,   t3,   v7     \n\t"
			
			"addi       a4,   a4,   16     \n\t"

			"bne        a4,   %[end], .matxvecvi32\n\t"
			"mv         a5,   %[o_addr]    \n\t"
			"vsw.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v1,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v2,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsw.v      v3,   (a5)         \n\t"
			::
			[end]"r"(x+d_x),
			[i_size]"r"(d_x*16),
			[i_addr]"r"(x),
			[w_addr]"r"(w+(j*d_x)),
			[b_addr]"r"(b+j),
			[o_addr]"r"(y+j):
			"a0","a1","a2","a3","a4","a5"
			,"t0","t1","t2","t3"
//			,"v0","v1","v2","v3","v4","v5","v6","v7","v8"
			);

	}
#endif

	return;
}

// y = w*x + b
void matxvec_v(int16_t *w, int16_t *b,
		int16_t *x, int16_t *y,
		int d_x, int d_y){

	if( d_x<1 || d_y<1 ){
		return;
	}

#ifdef NEON
	int i, j, k;
	int16_t *w_ptr, *b_ptr, *o_ptr;

	int out_q, out_r;
	int in_q, in_r;

	out_q = d_y / 16 * 16;
	in_q  = d_x / 8 * 8;
	in_r  = d_x % 8;

	for(j=0; j<out_q; j+=16){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"mov       r4,   %[x]          \n\t"
			"mov       r0,   %[w]          \n\t"
			"mov       r5,   %[b]          \n\t"
			"add       r1,   r0,   %[ws]   \n\t"
			"vmov.u32  q8,   #0            \n\t"
			"vmov.u32  q9,   #0            \n\t"
			"cmp       r5,   #0            \n\t"
			"beq       .matxvecvi16_1_L0   \n\t" // load biases to acc registers
			"vld1.16   {q8}, [r5]!         \n\t"
			"vld1.16   {q9}, [r5]!         \n\t"
		".matxvecvi16_1_L0:                \n\t" // set acc registers to 0
			"vmov.u32  q10,  #0            \n\t"
			"vmov.u32  q11,  #0            \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
		".matxvecvi16_1_L2:                \n\t"
			"vld1.16   {q0}, [r4]!         \n\t"
			"vmla.i16  q8,   q4,   d0[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[0]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"vmla.i16  q10,  q6,   d0[1]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d0[1]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"vmla.i16  q8,   q4,   d0[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[2]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"vmla.i16  q10,  q6,   d0[3]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d0[3]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"vmla.i16  q8,   q4,   d1[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d1[0]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"vmla.i16  q10,  q6,   d1[1]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d1[1]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"vmla.i16  q8,   q4,   d1[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d1[2]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"vmla.i16  q10,  q6,   d1[3]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d1[3]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"cmp       r4,   %[xend]       \n\t"
			"blt       .matxvecvi16_1_L2   \n\t"
			"cmp       %[xr],#0            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vld1.16   {q0}, [r4]!         \n\t"
			"vmla.i16  q8,   q4,   d0[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[0]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"cmp       %[xr],#1            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q10,  q6,   d0[1]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d0[1]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"cmp       %[xr],#2            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q8,   q4,   d0[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[2]   \n\t"
			"vld1.16   {q5}, [r1]!         \n\t"
			"cmp       %[xr],#3            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q10,  q6,   d0[3]   \n\t"
			"vld1.16   {q6}, [r0]!         \n\t"
			"vmla.i16  q11,  q7,   d0[3]   \n\t"
			"vld1.16   {q7}, [r1]!         \n\t"
			"cmp       %[xr],#4            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q8,   q4,   d1[0]   \n\t"
			"vld1.16   {q4}, [r0]          \n\t"
			"vmla.i16  q9,   q5,   d1[0]   \n\t"
			"vld1.16   {q5}, [r1]          \n\t"
			"cmp       %[xr],#5            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q10,  q6,   d1[1]   \n\t"
			"vmla.i16  q11,  q7,   d1[1]   \n\t"
			"cmp       %[xr],#6            \n\t"
			"beq       .matxvecvi16_1_L3   \n\t"
			"vmla.i16  q8,   q4,   d1[2]   \n\t"
			"vmla.i16  q9,   q5,   d1[2]   \n\t"
		".matxvecvi16_1_L3:                \n\t"
			"mov       r5,   %[y]          \n\t"
			"vadd.i16  q8,   q8,   q10     \n\t"
			"vadd.i16  q9,   q9,   q11     \n\t"
			"vst1.16   {q8}, [r5]!         \n\t"
			"vst1.16   {q9}, [r5]!         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[ws]"r"(d_x*16), // weight stride
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"r0","r1","r2","r3","r4","r5",
			"q0","q4","q5","q6","q7","q8","q9","q10","q11"
			);
	}
	out_q = d_y / 8 * 8;
	for(; j<out_q; j+=8){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"mov       r4,   %[x]          \n\t"
			"mov       r0,   %[w]          \n\t"
			"mov       r5,   %[b]          \n\t"
			"vmov.u32  q8,   #0            \n\t"
			"cmp       r5,   #0            \n\t"
			"beq       .matxvecvi16_2_L0   \n\t" // load biases to acc registers
			"vld1.16   {q8}, [r5]!         \n\t"
		".matxvecvi16_2_L0:                \n\t" // set acc registers to 0
			"vmov.u32  q9,   #0            \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"
		".matxvecvi16_2_L2:                \n\t"
			"vld1.16   {q0}, [r4]!         \n\t"
			"vmla.i16  q8,   q4,   d0[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[1]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"
			"vmla.i16  q8,   q4,   d0[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d0[3]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"
			"vmla.i16  q8,   q4,   d1[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d1[1]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"
			"vmla.i16  q8,   q4,   d1[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"
			"vmla.i16  q9,   q5,   d1[3]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"
			"cmp       r4,   %[xend]       \n\t"
			"blt       .matxvecvi16_2_L2   \n\t"

			"cmp       %[xr],#0            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vld1.16   {q0}, [r4]!         \n\t"
			"vmla.i16  q8,   q4,   d0[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"

			"cmp       %[xr],#1            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q9,   q5,   d0[1]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"

			"cmp       %[xr],#2            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q8,   q4,   d0[2]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"

			"cmp       %[xr],#3            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q9,   q5,   d0[3]   \n\t"
			"vld1.16   {q5}, [r0]!         \n\t"

			"cmp       %[xr],#4            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q8,   q4,   d1[0]   \n\t"
			"vld1.16   {q4}, [r0]!         \n\t"

			"cmp       %[xr],#5            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q9,   q5,   d1[1]   \n\t"

			"cmp       %[xr],#6            \n\t"
			"beq       .matxvecvi16_2_L3   \n\t"
			"vmla.i16  q8,   q4,   d1[2]   \n\t"

		".matxvecvi16_2_L3:                \n\t"
			"vadd.i16  q8,   q8,   q9      \n\t"
			"vst1.16   {q8}, [%[y]]        \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"r0","r1","r2","r3","r4","r5",
			"q0","q4","q5","q8","q9"
			);

	}

	out_r = d_y % 8;
	if(out_r==0) return;

	b_ptr = (b==0) ? 0 : (b+j);
	w_ptr = w+(j*d_x);
	o_ptr = y+j;
	matxvec(w_ptr, b_ptr, x, o_ptr, d_x, out_r);

#elif RISCV
	int i, j, k;
	int16_t *w_ptr, *b_ptr, *o_ptr;

	int out_q, out_r;
	int in_q, in_r;

	out_q = d_y / 16 * 16;
	in_q  = d_x / 8 * 8;
	in_r  = d_x % 8;

	for(j=0; j<out_q; j+=16){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;

		asm(
			"addi       a3,   zero, 999    \t\n"
			"vsetvli    zero, a3,   e16    \t\n"
			"mv         a4,   %[x]         \n\t"
			"mv         a0,   %[w]         \n\t"
			"mv         a5,   %[b]         \n\t"
			"add        a1,   a0,   %[ws]  \n\t"
			"vxor.vv    v8,   v8,   v8     \n\t"
			"vxor.vv    v9,   v9,   v9     \n\t"
			"beq        a5,   zero, .matxvecvi16_1_L0\n\t"
			"vlh.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vlh.v      v9,   (a5)         \n\t"
		".matxvecvi16_1_L0:                \n\t"
			"vxor.vv    v10,  v10,  v10    \n\t"
			"vxor.vv    v11,  v11,  v11    \n\t"
			"lh         t0,   0(a4)        \n\t"
			"lh         t1,   2(a4)        \n\t"
			"lh         t2,   4(a4)        \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
		".matxvecvi16_1_L2:                \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t3,   6(a4)        \n\t"
			
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t0,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t0,   8(a4)        \n\t"
			
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t1,   v7     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t1,   10(a4)       \n\t"
			
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t2,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t2,   12(a4)       \n\t"
			
			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t3,   v7     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t3,   14(a4)       \n\t"
			
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t0,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t0,   16(a4)       \n\t"
			
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t1,   v7     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t1,   18(a4)       \n\t"
			
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t2,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t2,   20(a4)       \n\t"
			"addi       a4,   a4,   16     \n\t"

			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t3,   v7     \n\t"
			
			"blt        a4,   %[xend], .matxvecvi16_1_L2\n\t"

			"mv         a5,   %[xr]        \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t3,   6(a4)        \n\t"
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t0,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t0,   8(a4)        \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t1,   v7     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t1,   10(a4)       \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t2,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			"lh         t2,   12(a4)       \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v11,  t3,   v7     \n\t"
			"vlh.v      v7,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vmacc.vx   v9,   t0,   v5     \n\t"
			"vlh.v      v5,   (a1)         \n\t"
			"addi       a1,   a1,   16     \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vmacc.vx   v11,  t1,   v7     \n\t"
			
			"beq        a5,   zero, .matxvecvi16_1_L3\n\t"
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vmacc.vx   v9,   t2,   v5     \n\t"
		".matxvecvi16_1_L3:                \n\t"
			"vadd.vv    v8,   v8,   v10    \n\t"
			"vadd.vv    v9,   v9,   v11    \n\t"
			"mv         a5,   %[y]         \n\t"
			"vsh.v      v8,   (a5)         \n\t"
			"addi       a5,   a5,   16     \n\t"
			"vsh.v      v9,   (a5)         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[ws]"r"(d_x*16), // weight stride
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"a0","a1","a2","a3","a4","a5"
			,"t0","t1","t2","t3"
//			,"v0","v1","v2","v3","v4","v5","v6","v7","v8"
			);
	}
	
	out_q = d_y / 8 * 8;
	for(; j<out_q; j+=8){

		b_ptr = (b==0) ? 0 : (b+j);
		w_ptr = w+(j*d_x);
		o_ptr = y+j;
		asm(
			"addi       a3,   zero, 999    \t\n"
			"vsetvli    zero, a3,   e16    \t\n"
			"mv         a4,   %[x]         \n\t"
			"mv         a0,   %[w]         \n\t"
			"mv         a5,   %[b]         \n\t"
			"vxor.vv    v8,   v8,   v8     \n\t"
			"beq        a5,   zero, .matxvecvi16_2_L0\n\t"
			"vlh.v      v8,   (a5)         \n\t"
		".matxvecvi16_2_L0:                \n\t"
			"vxor.vv    v10,  v10,  v10    \n\t"
			"lh         t0,   0(a4)        \n\t"
			"lh         t1,   2(a4)        \n\t"
			"lh         t2,   4(a4)        \n\t"
			"lh         t3,   6(a4)        \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
		".matxvecvi16_2_L2:                \n\t"

			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t0,   8(a4)        \n\t"
			
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t1,   10(a4)       \n\t"
			
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t2,   12(a4)       \n\t"
			
			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t3,   14(a4)       \n\t"
			
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t0,   16(a4)       \n\t"
			
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t1,   18(a4)       \n\t"
			
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t2,   20(a4)       \n\t"
			
			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t3,   22(a4)        \n\t"
			"addi       a4,   a4,   16     \n\t"

			"blt        a4,   %[xend], .matxvecvi16_2_L2\n\t"
			"mv         a5,   %[xr]        \n\t"

			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t0,   8(a4)        \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t1,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t1,   10(a4)       \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v8,   t2,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			"lh         t2,   12(a4)       \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t3,   v6     \n\t"
			"vlh.v      v6,   (a0)         \n\t"
			"addi       a0,   a0,   16     \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v8,   t0,   v4     \n\t"
			"vlh.v      v4,   (a0)         \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"addi       a5,   a5,   -1     \n\t"
			"vmacc.vx   v10,  t1,   v6     \n\t"
			
			"beq        a5,   zero, .matxvecvi16_2_L3\n\t"
			"vmacc.vx   v8,   t2,   v4     \n\t"
		".matxvecvi16_2_L3:                \n\t"
			"vadd.vv    v8,   v8,   v10    \n\t"
			"mv         a5,   %[y]         \n\t"
			"vsh.v      v8,   (a5)         \n\t"
			::
			[xend]"r"(x+in_q),
			[xr]"r"(in_r),   // x remain
			[x]"r"(x),
			[w]"r"(w_ptr),
			[b]"r"(b_ptr),
			[y]"r"(o_ptr):
			"a0","a1","a2","a3","a4","a5"
			,"t0","t1","t2","t3"
//			,"v0","v1","v2","v3","v4","v5","v6","v7","v8"
			);
	}
	
	out_r = d_y % 8;
	if(out_r==0) return;

	b_ptr = (b==0) ? 0 : (b+j);
	w_ptr = w+(j*d_x);
	o_ptr = y+j;
	matxvec(w_ptr, b_ptr, x, o_ptr, d_x, out_r);

#endif

	return;
}

// y = w*x + b
void matxvec_v(int8_t *w, int8_t *b,
		int8_t *x, int8_t *y,
		int d_x, int d_y){

	if( d_x<1 || d_y<1 ){
		return;
	}

#ifdef NEON
	int i, j, k;
	for(j=0; j<d_y; j+=16){

		asm(
			"mov r4,%[i_addr]\n\t"
			"mov r0,%[w_addr]\n\t"
			"mov r5,%[b_addr]\n\t"
			"vld1.8 {q8}, [r5]!\n\t"
			"veor   q8, q8, q8\n\t"
			"veor   q9, q9, q9\n\t"

		".matxvecvi8:\n\t"
			"vld1.8   {q0}, [r4]!      \n\t"
			

			"cmp r4, %[end]\n\t"
			"bne .matxvecvi8\n\t"
			"mov r5,%[o_addr]\n\t"
			"vst1.8 {q8},[r5]!\n\t"
			::
			[end]"r"(x+d_x),
			[i_addr]"r"(x),
			[w_addr]"r"(w+(j*d_x)),
			[b_addr]"r"(b+j),
			[o_addr]"r"(y+j):
			"r0","r1","r4","r5",
			"q0","q1","q2","q3","q4","q5","q6","q7",
			"q8","q9","q10","q11","q12","q13","q14","q15"
			);

	}
#elif RISCV
#endif

	return;
}
