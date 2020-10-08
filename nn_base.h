#include <vector>
#include <cstring>
#include <iostream>
#include "mymodule_vec.h"
#include "mymodule.h"
#include <cmath>

using namespace std;

template <typename Elem>
class FcLayer {

public:

	int n_input_channel;
	int n_output_channel;
	Elem* w;
	Elem* b;
	
	FcLayer(int n_in, int n_out){
		n_input_channel  = n_in;
		n_output_channel = n_out;
		w = (Elem*)malloc(sizeof(Elem) * n_input_channel * n_output_channel);
		b = (Elem*)malloc(sizeof(Elem) * n_output_channel);
	}

	~FcLayer(){
		if(w) free(w);
		if(b) free(b);
	}


	void forward(Elem* in, Elem* out) {
		matxvec_v(this->w, this->b, in, out,
				  n_input_channel, n_output_channel);	
	}

	void set_weights(Elem* w) {
		memcpy(this->w, w, sizeof(Elem) * n_input_channel * n_output_channel);
	}

	void set_biases(Elem* b) {
		memcpy(this->b, b, sizeof(Elem) * n_output_channel);
	}

	void set_params(Elem* w, Elem* b) {
		set_weights(w);		
		set_biases(b);		
	}

	void set_params(const char* w_fn, const char* b_fn) {

		FILE *fp;
		fp = fopen(w_fn, "r");
		fread(w, sizeof(Elem), n_input_channel*n_output_channel , fp);
		fclose(fp);

		if(*b_fn == 0){
			free(b);
			b = 0;
		}
		else{
			fp = fopen(b_fn, "r");
			fread(b, sizeof(Elem), n_output_channel , fp);
			fclose(fp);
		}
	}

};

template <typename Elem>
class FcLayerSVD{

public:
	int n_input_channel;
	int n_output_channel;
	Elem* w;
	Elem* b;
	int rank;
	Elem* w2;
	Elem* buf;
	
	FcLayerSVD(int n_in, int n_out, int rank_): FcLayer<Elem>(n_in, n_out) {
		n_input_channel  = n_in;
		n_output_channel = n_out;
		w = (Elem*)malloc(sizeof(Elem) * n_input_channel * n_output_channel);
		b = (Elem*)malloc(sizeof(Elem) * n_output_channel);
		rank = rank_;
		w2  = (Elem*)malloc(sizeof(Elem) * n_output_channel * rank);
		buf = (Elem*)malloc(sizeof(Elem) * rank);
	}

	~FcLayerSVD(){
		if(w) free(w);
		if(b) free(b);
		if(w2) free(w2);
		if(buf) free(buf);
	}


	void forward(Elem* in, Elem* out) {
		matxvec_v(w, 0, in, buf,
				  n_input_channel, rank);	
		matxvec_v(w2, b, buf, out,
				  rank, n_output_channel);	
	}

	void set_weights(Elem* w1, Elem* w2) {
		memcpy(w,  w1, sizeof(Elem) * n_input_channel * rank);
		memcpy(w2, w2, sizeof(Elem) * rank * n_output_channel);
	}

};


template <typename Elem>
class LSTM {

public:
//private:
	int n_input;
	int n_hidden;
	Elem* w_ih;
	Elem* w_hh;
	Elem* b;
	Elem* buf1;
	Elem* buf2;

public:
	
	LSTM(int n_input, int n_hidden){
		this->n_input  = n_input;
		this->n_hidden = n_hidden;
		w_ih = (Elem*)malloc(sizeof(Elem) * n_input  * n_hidden * 4);
		w_hh = (Elem*)malloc(sizeof(Elem) * n_hidden * n_hidden * 4);
		b    = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);
		buf1 = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);
		buf2 = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);
	}

	~LSTM(){
		if(w_ih) free(w_ih);
		if(w_hh) free(w_hh);
		if(b) free(b);
		if(buf1) free(buf1);
		if(buf2) free(buf2);
	}


	void forward(
			Elem* in,
			Elem* h_in, Elem* h_out,
			Elem* c_in, Elem* c_out) {

		matxvec_v(w_ih, b,    in,   buf1, n_input,  n_hidden*4);	
		matxvec_v(w_hh, buf1, h_in, buf2, n_hidden, n_hidden*4);
		
		sigmoid(buf2, 2*n_hidden);
		array_tanh(buf2+2*n_hidden, n_hidden);
		sigmoid(buf2+3*n_hidden, n_hidden);

		int i;

		for(i=0; i<n_hidden; i++){
			c_out[i] = c_in[i]*buf2[i+n_hidden]
				+ buf2[i]*buf2[i+2*n_hidden];
			h_out[i] = c_out[i];
		}
		array_tanh(h_out, n_hidden);
		for(i=0; i<n_hidden; i++){
			h_out[i] *= buf2[i+3*n_hidden];
		}

	}

	void set_params(Elem* w_ih, Elem* w_hh, Elem* b) {
		memcpy(this->w_ih, w_ih, sizeof(Elem)*n_input*n_hidden*4);
		memcpy(this->w_hh, w_hh, sizeof(Elem)*n_hidden* n_hidden);
		memcpy(this->b, b, sizeof(Elem)*n_hidden*4);
	}

};

template <typename Elem>
class LSTMSVD {

public:
//private:
	int n_input;
	int n_hidden;
	int ih_rank;
	int hh_rank;
	Elem* w_ih_r;
	Elem* w_ih_l;
	Elem* w_hh_r;
	Elem* w_hh_l;
	Elem* b;
	Elem* buf1;
	Elem* buf2;
	Elem* buf3;

public:
	
	LSTMSVD(int n_input, int n_hidden, int ih_rank, int hh_rank){
		this->n_input  = n_input;
		this->n_hidden = n_hidden;
		this->ih_rank = ih_rank;
		this->hh_rank = hh_rank;
		w_ih_r = (Elem*)malloc(sizeof(Elem) * n_input  * ih_rank);
		w_ih_l = (Elem*)malloc(sizeof(Elem) * ih_rank * n_hidden * 4);
		w_hh_r = (Elem*)malloc(sizeof(Elem) * n_hidden * hh_rank);
		w_hh_l = (Elem*)malloc(sizeof(Elem) * hh_rank * n_hidden * 4);
		b    = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);
		buf1 = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);
		buf2 = (Elem*)malloc(sizeof(Elem) * n_hidden * 4);

		int rank_max = ih_rank > hh_rank ? ih_rank : hh_rank;
		buf3 = (Elem*)malloc(sizeof(Elem) * rank_max);
	}

	~LSTMSVD(){
		if(w_ih_l) free(w_ih_l);
		if(w_ih_r) free(w_ih_r);
		if(w_hh_l) free(w_hh_l);
		if(w_hh_r) free(w_hh_r);
		if(b) free(b);
		if(buf1) free(buf1);
		if(buf2) free(buf2);
		if(buf3) free(buf3);
	}


	void forward(
			Elem* in,
			Elem* h_in, Elem* h_out,
			Elem* c_in, Elem* c_out) {

		//printf("lstm 1\n");
		matxvec_v(w_ih_r, 0,    in,   buf3, n_input,  ih_rank);	
		//printf("lstm 2\n");
		matxvec_v(w_ih_l, b,    buf3, buf1, ih_rank,  4*n_hidden);
		//printf("lstm 3\n");
		matxvec_v(w_hh_r, 0,    h_in, buf3, n_hidden, hh_rank);
		//printf("lstm 4\n");
		matxvec_v(w_hh_l, buf1, buf3, buf2, hh_rank,  4*n_hidden);
		//printf("lstm 5\n");

/*
		sigmoid(buf2, 2*n_hidden);
		array_tanh(buf2+2*n_hidden, n_hidden);
		sigmoid(buf2+3*n_hidden, n_hidden);

		int i;

		for(i=0; i<n_hidden; i++){
			c_out[i] = c_in[i]*buf2[i+n_hidden]
				+ buf2[i]*buf2[i+2*n_hidden];
			h_out[i] = c_out[i];
		}
		array_tanh(h_out, n_hidden);
		for(i=0; i<n_hidden; i++){
			h_out[i] *= buf2[i+3*n_hidden];
		}
*/
	}

	void set_params(
			Elem* w_ih_r, Elem* w_ih_l,
			Elem* w_hh_r, Elem* w_hh_l,
			Elem* b) {
		memcpy(this->w_ih_r, w_ih_r, sizeof(Elem)*n_input*ih_rank   );
		memcpy(this->w_ih_l, w_ih_l, sizeof(Elem)*ih_rank*n_hidden*4);
		memcpy(this->w_hh_r, w_hh_r, sizeof(Elem)*n_hidden*hh_rank  );
		memcpy(this->w_hh_l, w_hh_l, sizeof(Elem)*hh_rank*n_hidden*4);
		memcpy(this->b     , b     , sizeof(Elem)*n_hidden*4        );
	}

	void set_params(
			const char* w_ih_r_fn,
			const char* w_ih_l_fn,
			const char* w_hh_r_fn,
			const char* w_hh_l_fn,
			const char* b_fn     ) {

		FILE *fp;
		fp = fopen(w_ih_r_fn, "r");
		fread(w_ih_r, sizeof(Elem), n_input*ih_rank   , fp);
		fclose(fp);

		fp = fopen(w_ih_l_fn, "r");
		fread(w_ih_l, sizeof(Elem), ih_rank*n_hidden*4, fp);
		fclose(fp);
		
		fp = fopen(w_hh_r_fn, "r");
		fread(w_hh_r, sizeof(Elem), n_hidden*hh_rank  , fp);
		fclose(fp);
		
		fp = fopen(w_hh_l_fn, "r");
		fread(w_hh_l, sizeof(Elem), hh_rank*n_hidden*4, fp);
		fclose(fp);
		
		fp = fopen(b_fn, "r");
		fread(b     , sizeof(Elem), n_hidden*4        , fp);
		fclose(fp);

	}

};


template <typename Elem>
class Encoder {

public:
//private:

	LSTMSVD<Elem> *lstm0;
	LSTMSVD<Elem> *lstm1;
	LSTMSVD<Elem> *lstm2;
	LSTMSVD<Elem> *lstm3;
	LSTMSVD<Elem> *lstm4;
	int n_hidden;
	int n_input;
	Elem* lstm0_out;
	Elem* lstm1_out;
	Elem* lstm2_out;
	Elem* lstm3_out;
	Elem* lstm4_out;
	
	Elem* lstm_cin;
	Elem* lstm_cout;
	Elem* lstm_hin;
	Elem* lstm_hout;
	int maxlen=1024;
	int rank_ih=20;
	int rank_hh=256;

public:
	
	Encoder(){
		n_hidden = 1024;
		n_input  = 83;
		lstm0 = new LSTMSVD<Elem>(n_input,  n_hidden, rank_ih, rank_hh);
		lstm1 = new LSTMSVD<Elem>(n_hidden, n_hidden, rank_hh, rank_hh);
		lstm2 = new LSTMSVD<Elem>(n_hidden, n_hidden, rank_hh, rank_hh);
		lstm3 = new LSTMSVD<Elem>(n_hidden, n_hidden, rank_hh, rank_hh);
		lstm4 = new LSTMSVD<Elem>(n_hidden, n_hidden, rank_hh, rank_hh);
		lstm0_out = (Elem*)malloc(sizeof(Elem) * n_hidden * maxlen);
		lstm1_out = (Elem*)malloc(sizeof(Elem) * n_hidden * maxlen/2);
		lstm2_out = (Elem*)malloc(sizeof(Elem) * n_hidden * maxlen/4);
		lstm3_out = (Elem*)malloc(sizeof(Elem) * n_hidden * maxlen/8);
		lstm4_out = (Elem*)malloc(sizeof(Elem) * n_hidden * maxlen/16);
		
		lstm_cin  = (Elem*)malloc(sizeof(Elem) * n_hidden);
		lstm_cout = (Elem*)malloc(sizeof(Elem) * n_hidden);
		lstm_hin  = (Elem*)malloc(sizeof(Elem) * n_hidden);
		lstm_hout = (Elem*)malloc(sizeof(Elem) * n_hidden);

	}

	~Encoder(){
		delete lstm0;
		delete lstm1;
		delete lstm2;
		delete lstm3;
		delete lstm4;
		free(lstm0_out);
		free(lstm1_out);
		free(lstm2_out);
		free(lstm3_out);
		free(lstm4_out);
		free(lstm_cin );
		free(lstm_cout);
		free(lstm_hin );
		free(lstm_hout);
	}

	void forward(Elem* input, Elem* output, int len){

		int i;

		Elem *inptr, *hinptr, *houtptr, *cinptr, *coutptr;
		Elem* temp;
	// LSTM layer 0
		printf("encoder layer 0 start\n");
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			lstm_hin[i] = 0;
		}
		hinptr  = lstm_hin;
		cinptr  = lstm_cin;
		coutptr = lstm_cout;
		for(i=0; i<len; i++){
			inptr   = input + n_input*i;
			houtptr = lstm0_out + n_hidden*i;

			lstm0->forward(inptr, hinptr, houtptr, cinptr, coutptr);

			hinptr = houtptr;
			temp = cinptr;
			cinptr = coutptr;
			coutptr = temp;
		}
		//for(i=0; i<16; i++){printf("%6.3f ",lstm0_out[i]);} printf("\n\n");
		max_pooling(lstm0_out, n_hidden, len);

	// LSTM layer 1
		printf("encoder layer 1 start\n");
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			//lstm_hin[i] = 0;
		}
		hinptr  = lstm_hin;
		cinptr  = lstm_cin;
		coutptr = lstm_cout;
		for(i=0; i<len/2; i++){
			inptr   = lstm0_out + n_hidden*i;
			houtptr = lstm1_out + n_hidden*i;

			lstm1->forward(inptr, hinptr, houtptr, cinptr, coutptr);

			hinptr = houtptr;
			temp = cinptr;
			cinptr = coutptr;
			coutptr = temp;
		}
		max_pooling(lstm1_out, n_hidden, (len+1)/2);
		
	// LSTM layer 2
		printf("encoder layer 2 start\n");
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			//lstm_hin[i] = 0;
		}
		hinptr  = lstm_hin;
		cinptr  = lstm_cin;
		coutptr = lstm_cout;
		for(i=0; i<len/4; i++){
			inptr   = lstm1_out + n_hidden*i;
			houtptr = lstm2_out + n_hidden*i;

			lstm2->forward(inptr, hinptr, houtptr, cinptr, coutptr);

			hinptr = houtptr;
			temp = cinptr;
			cinptr = coutptr;
			coutptr = temp;
		}
		max_pooling(lstm2_out, n_hidden, (len+3)/4);
		
	// LSTM layer 3
		printf("encoder layer 3 start\n");
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			//lstm_hin[i] = 0;
		}
		hinptr  = lstm_hin;
		cinptr  = lstm_cin;
		coutptr = lstm_cout;
		for(i=0; i<len/8; i++){
			inptr   = lstm2_out + n_hidden*i;
			houtptr = lstm3_out + n_hidden*i;

			lstm3->forward(inptr, hinptr, houtptr, cinptr, coutptr);

			hinptr = houtptr;
			temp = cinptr;
			cinptr = coutptr;
			coutptr = temp;
		}
		max_pooling(lstm3_out, n_hidden, (len+7)/8);
		
	// LSTM layer 4
		printf("encoder layer 4 start\n");
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			//lstm_hin[i] = 0;
		}
		hinptr  = lstm_hin;
		cinptr  = lstm_cin;
		coutptr = lstm_cout;
		for(i=0; i<len/16; i++){
			inptr   = lstm3_out + n_hidden*i;
			//houtptr = lstm4_out + n_hidden*i;
			houtptr = output + n_hidden*i;

			lstm4->forward(inptr, hinptr, houtptr, cinptr, coutptr);

			hinptr = houtptr;
			temp = cinptr;
			cinptr = coutptr;
			coutptr = temp;
		}

	}

};

template <typename Elem>
class Energy {

public:
//private:

	int n_enc_h;
	int n_dec_h;
	int n_attn;

	//FcLayer<Elem> *W_enc;
	//FcLayer<Elem> *V_dec;
	Elem* w_enc;
	Elem* w_dec;
	Elem* b;
	Elem* v;
	Elem r;

	Elem* buf1;
	Elem* buf2;

public:
	
	Energy(int _n_enc_h, int _n_dec_h, int _n_attn){
		n_enc_h = _n_enc_h;
		n_dec_h = _n_dec_h;
		n_attn   = _n_attn;

		r = 1;
		//W_enc = new FcLayer<Elem>(n_enc_h, n_att);
		//V_dec = new FcLayer<Elem>(n_dec_h, n_att);
		w_enc = (Elem*)malloc(sizeof(Elem) * n_attn * n_enc_h);
		w_dec = (Elem*)malloc(sizeof(Elem) * n_attn * n_dec_h);
		b = (Elem*)malloc(sizeof(Elem) * n_attn);
		v = (Elem*)malloc(sizeof(Elem) * n_attn);
		buf1 = (Elem*)malloc(sizeof(Elem) * n_attn);
		buf2 = (Elem*)malloc(sizeof(Elem) * n_attn);
	}

	~Energy(){
		//delete W_enc;
		//delete V_dec;
		free(w_enc);
		free(w_dec);
		free(b);
		free(v);
		free(buf1);
		free(buf2);
	}

	float forward(Elem* enc_h, Elem* dec_h){
		
		matxvec_v(w_enc, b,    enc_h, buf1, n_enc_h, n_attn);	
		matxvec_v(w_dec, buf1, dec_h, buf2, n_dec_h, n_attn);

		array_tanh(buf2, n_attn);
		
		Elem sum;
		sum = r;
		for(int i=0; i<n_attn; i++){
			sum += buf2[i] * v[i];
		}

		return sum;
	}

	void set_params(
			const char* w_enc_fn,
			const char* w_dec_fn,
			const char* b_fn,
			const char* v_fn,
			const char* r_fn
			) {
		
		FILE *fp;
		fp = fopen(w_enc_fn, "r");
		fread(w_enc, sizeof(Elem), n_enc_h*n_attn, fp);
		fclose(fp);

		fp = fopen(w_dec_fn, "r");
		fread(w_dec, sizeof(Elem), n_dec_h*n_attn, fp);
		fclose(fp);
		
		fp = fopen(b_fn, "r");
		fread(b, sizeof(Elem), n_attn, fp);
		fclose(fp);
		
		fp = fopen(v_fn, "r");
		fread(v, sizeof(Elem), n_attn, fp);
		fclose(fp);
		
		fp = fopen(r_fn, "r");
		fread(&r, sizeof(Elem), 1, fp);
		fclose(fp);
	}

};

template <typename Elem>
class Decoder {

public:
//private:

	int n_hidden;
	int n_vocab;
	int n_emb;
	int n_out;
	int n_att;
	int lstm_rank;
	LSTMSVD<Elem> *lstm;
	FcLayer<Elem> *embedding;
	FcLayer<Elem> *combine;
	FcLayer<Elem> *proj_vocab;
	Energy<Elem> *enc_energy;
	Energy<Elem> *mono_energy;

	Elem* buf1;
	Elem* buf2;
	Elem* buf3;
	Elem* buf4;
	Elem* buf5;
	Elem* buf6;
	Elem* buf7;
	Elem* buf8;

	int chunk_size;

public:
	
	Decoder(){
		n_hidden = 1024;
		n_vocab = 10001;
		n_emb = 1024;
		n_out = 1024;
		n_att = 1024;
		chunk_size = 3;
		lstm_rank = 256;
		
		lstm = new LSTMSVD<Elem>(n_emb, n_hidden, lstm_rank, lstm_rank);
		embedding  = new FcLayer<Elem>(n_vocab, n_emb);
		combine    = new FcLayer<Elem>(n_hidden*2, n_out);
		proj_vocab = new FcLayer<Elem>(n_out, n_vocab);
		enc_energy  = new Energy<Elem>(n_hidden, n_hidden, n_att);
		mono_energy = new Energy<Elem>(n_hidden, n_hidden, n_att);
		
		buf1 = (Elem*)malloc(sizeof(Elem) * n_hidden);
		buf2 = (Elem*)malloc(sizeof(Elem) * n_hidden);
		buf3 = (Elem*)malloc(sizeof(Elem) * n_hidden);
		buf4 = (Elem*)malloc(sizeof(Elem) * n_hidden);
		buf5 = (Elem*)malloc(sizeof(Elem) * n_hidden);
		buf6 = (Elem*)malloc(sizeof(Elem) * n_hidden*2);
		buf7 = (Elem*)malloc(sizeof(Elem) * n_out);
		buf8 = (Elem*)malloc(sizeof(Elem) * n_vocab);

	}

	~Decoder(){
		delete lstm       ; 
		delete embedding  ; 
		delete combine    ; 
		delete proj_vocab ; 
		delete enc_energy ; 
		delete mono_energy; 

		free(buf1);
		free(buf2);
		free(buf3);
		free(buf4);
		free(buf5);
		free(buf6);
		free(buf7);
		free(buf8);
	}

	int forward(Elem* input, int* output, int len, int max_out_len){

		int i, j;
	
		Elem *lstm_cin, *lstm_cout;
		Elem *lstm_hin, *lstm_hout;
		Elem *e_enc_h, *e_dec_h; 
		Elem *context;
		Elem *concat_in;
		Elem *out;
		Elem *word_vec;
		Elem *word_vec_out;
		Elem *temp;

		lstm_cin  = buf1;  
		lstm_cout = buf2;
		lstm_hin  = buf3;
		lstm_hout = buf4;
		context   = buf5;
		concat_in = buf6;
		out       = buf7;
		word_vec_out = buf8;
		for(i=0; i<n_hidden; i++){
			lstm_cin[i] = 0;
			lstm_hin[i] = 0;
		}

		int t=0;
		int k;
		float u[3];
		int word;
		float p_th;
		float p;
		float e;
		p_th = 0.5;
		word = 9999;
		for(i=0; i<max_out_len; i++){
			
			word_vec = embedding->w + word*n_emb;
			lstm->forward(word_vec, lstm_hin, lstm_hout, lstm_cin, lstm_cout);

			for(j=t; j<len; j++){
				e_enc_h = input + j*n_hidden;
				e_dec_h = lstm_hout;

				e = mono_energy->forward(e_enc_h, e_dec_h);
				p = 1.0 / (1.0 + expf(-e)); // p = sigmoid(e);
				if(p >= p_th || j==(len-1)){
					for(k=0; k<chunk_size; k++){
						int enc_idx;
						enc_idx = j-(chunk_size-1-k);
						if(enc_idx<0) {
							continue;
						}
						else {
							e_enc_h = input + enc_idx*n_hidden;
							u[k] = enc_energy->forward(e_enc_h, e_dec_h);
						}
					}
					softmax(u, chunk_size);

					for(k=0; k<n_hidden; k++){
						context[k] = 0;
					}

					for(k=0; k<chunk_size; k++){
						int enc_idx;
						enc_idx = j-(chunk_size-1-k);
						if(enc_idx<0) {
							continue;
						}
						else {
							e_enc_h = input + enc_idx*n_hidden;
							for(int ii=0; ii<n_hidden; ii++){
								context[ii] += (e_enc_h[ii] * u[k]);
							}
						}
					}
					
					t = j;
					break;
				}

			}

			for(k=0; k<n_hidden; k++){
				concat_in[k] = context[k];
			}
			for(k=0; k<n_hidden; k++){
				concat_in[k+n_hidden] = lstm_hout[k];
			}

			combine->forward(concat_in, out);
			array_tanh(out, n_out);
			proj_vocab->forward(out, word_vec_out);
			word = max(word_vec_out, n_vocab);
			output[i] = word;
			temp      = lstm_cin;
			lstm_cin  = lstm_cout;  
			lstm_cout = temp;
			temp      = lstm_hin;
			lstm_hin  = lstm_hout;  
			lstm_hout = temp;
		

			if(word==10000){
				break;
			}

		}

		return (i+1) < max_out_len ? (i+1) : max_out_len;

	}

};



template <typename Elem>
class MoChA {

public:
//private:
	Encoder<Elem> *enc;
	Decoder<Elem> *dec;
	int n_h;
	int n_i_max;
	Elem* enc_output;

public:
	
	MoChA(){
		n_h = 1024;
		n_i_max = 1024;
		enc = new Encoder<Elem>();
		dec = new Decoder<Elem>();
		enc_output = (Elem*)malloc(sizeof(Elem) * n_h * n_i_max/16);
	}

	~MoChA(){
		delete enc;
		delete dec;
		free(enc_output);
	}

	int run(Elem* input, int* output, int in_len, int max_out_len){


		printf("encoder start\n");
		enc->forward(input, enc_output, in_len);

		printf("decoder start\n");
		int out_len;
		out_len = dec->forward(enc_output, output, (in_len+15)/16, max_out_len);

		return out_len;
	}

	int run(Elem* input, int* output, int in_len, int max_out_len, int debug){

		asm("L_KJH_0:\n\t":::);
		if(debug == 0) return 0;

		printf("encoder start\n");
		enc->forward(input, enc_output, in_len);

		asm("L_KJH_1:\n\t":::);
		if(debug == 1) return 0;

		printf("decoder start\n");
		int out_len;
		out_len = dec->forward(enc_output, output, (in_len+15)/16, max_out_len);
		
		asm("L_KJH_2:\n\t":::);

		return out_len;
	}

};

