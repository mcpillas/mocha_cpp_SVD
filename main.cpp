#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <unistd.h>           // needed for pid_t type
#include <string.h>

#include "mymodule.h"
#include "mymodule_vec.h"
#include "nn_base.h"

typedef float   elem_t;
//typedef int32_t elem_t;
//typedef int16_t elem_t;
//typedef int8_t  elem_t;


int main(int argc, char* argv[]){

	int i;
	
	int n_input     = 83;
	int n_hidden    = 1024;
	int n_words     = 10001;
	int n_embed     = 1024;
	int n_attn      = 256;
	int n_dec_out   = 1024;
	int n_rank1     = 24;   // 30% of 83
	int n_rank2     = 307;  // 30% of 1024


// create model
	MoChA<elem_t> *mocha = new MoChA<elem_t>();

	mocha->enc->lstm0->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_0_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_0_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_0_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_0_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_0_bias.bin"
			);
	mocha->enc->lstm1->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_1_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_1_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_1_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_1_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_1_bias.bin"
			);
	mocha->enc->lstm2->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_2_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_2_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_2_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_2_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_2_bias.bin"
			);
	mocha->enc->lstm3->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_3_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_3_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_3_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_3_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_3_bias.bin"
			);
	mocha->enc->lstm4->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_4_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_4_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_4_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_4_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/enc_lstm_4_bias.bin"
			);
	mocha->dec->lstm->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_lstm_weight_ih_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_lstm_weight_ih_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_lstm_weight_hh_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_lstm_weight_hh_u.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_lstm_bias.bin"
			);
	mocha->dec->combine->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_combine_weight.bin",
			""
			);
	mocha->dec->proj_vocab->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_proj_vocab.bin",
			""
			);
	mocha->dec->enc_energy->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_enc_energy_w_enc.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_enc_energy_w_dec.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_enc_energy_b.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_enc_energy_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_enc_energy_r.bin"
			);
	mocha->dec->mono_energy->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_mono_energy_w_enc.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_mono_energy_w_dec.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_mono_energy_b.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_mono_energy_v.bin",
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_mono_energy_r.bin"
			);
	mocha->dec->embedding->set_params(
			"/home/postech/examples/mocha_cpp_SVD_v9/model/SVD_25/dec_embedding.bin",
			""
			);


// Load input data
	elem_t* input;
	int* output;

	int in_len      = 531;
	int max_out_len = 22;
	int out_len;

	input  = (elem_t*)malloc(sizeof(elem_t)*in_len*n_input);

	FILE *fp;
	fp = fopen("/home/postech/examples/mocha_cpp_SVD_v9/feature/SVD_25_feature/input_tensor.bin", "r");
	fread(input, sizeof(elem_t), in_len*n_input, fp); fclose(fp);
	output = (int*)malloc(sizeof(int) * max_out_len);

	printf("debug: main.cpp 1\n");

// run MoChA
	//out_len = mocha->run(input, output, in_len, max_out_len);
	int debug_opt;
	debug_opt = (argc > 1) ? atoi(argv[1]) : 3;
	out_len = mocha->run(input, output, in_len, max_out_len, debug_opt);

	printf("debug: main.cpp 2 %d\n");
	printf("debug: in_len=%d, out_len=%d\n", in_len, out_len);

// print result
	printf("output: \n");
	for(i=0; i<out_len; i++){
		printf("%d ", output[i]);
	}
	printf("\n\n");

	free(input);
	free(output);

	return 0;
}
