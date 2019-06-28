/*メモ
プログラムをいじったので、もう一度害虫を復元できるか確認
それができたら葉の復元を確認
それもできたらユニット数増強と学習係数の調整
最終的には重みの保存までいきたい
バイアスhがおかC
特徴抽出層の最後の出力をnormalizeした後に、一般的な三層のNNに入力する。活性化関数はsigmoid関数
normalized_arrなどを作成して、それを三層の入力にあてるのがいいと考えた
device_arr_hidden[level][m][n]が出力なので、これを利用する
[8/3]最下層のNNの値のセット準備完了。転送命令を書くこと。
[8/4]転送完了。計算と逆転送命令を書くこと。
[8/6]バイアスは0.0001fにセット
*/
#include"opencv2/highgui/highgui.hpp"
#include"opencv2/core/core.hpp"
#include"opencv2/imgproc/imgproc.hpp"
#include<iostream>
#include<vector>
#include<string>
#include<cstdlib>
#include<random>
#include<cmath>
#include<fstream>
#include<iomanip>
using namespace std;
using namespace cv;
void device_ArrErrChk(cudaError_t);
void host_check_pic_insect();
void host_get_data_insect();
void host_check_pic_leaves();
void host_get_data_leaves();
void host_get_noisy_data_insect(int);
void host_get_noisy_data_leaves(int);
void host_set_value_to_weight_f();
void host_set_value_to_weight_s();
void host_set_value_to_RGBweight_f();
void host_set_value_to_RGBweight_s();
void host_set_value_to_bias_h();
void host_set_value_to_bias_o();
void host_set_value_to_RGBbias_o();
void host_set_value_to_last_layer_wf();
void host_set_value_to_last_layer_ws();
void host_set_value_to_last_layer_bias_f();
void host_set_value_to_last_layer_bias_s();
void host_calc_sum_RGBh(int);
__global__ void device_clear_arr();
__global__ void device_clear_arr_ll();
__global__ void device_calc_sum_RGBh(int);
__global__ void device_add_bias_RGBh();
void host_calc_sum_RGBo();
__global__ void device_calc_sum_RGBo();
__global__ void device_add_bias_RGBo();
void ReLu_h(int,const int,const int);
__global__ void device_ReLu_h(int const);
void ReLu_RGBo();
__global__ void device_ReLu_RGBo();
void training_RGBweight_s(int,const float);
__global__ void device_training_RGBweight_s(int,const float);
void training_RGBbias_s(int,const float);
__global__ void device_training_RGBbias_s(int,const float);
void training_RGBweight_f(int,const float);
__global__ void device_training_RGBweight_f(int,const float);
void training_RGBbias_f(int,const float);
__global__ void device_training_RGBbias_f(int,const float);
void show_input_with_window();
void show_results(int);
void save_weights_with_image();
void save_output_result_with_image(int);
void clear_arr(int);
void get_scaleRGB();
void host_arr_RGB_to_device();
void host_arr_weight_f_to_device();
void host_arr_weight_s_to_device();
void host_arr_RGB_weight_f_to_device();
void host_arr_RGB_weight_s_to_device();
void host_arr_bias_h_to_device();
void host_arr_bias_o_to_device();
void host_arr_RGB_bias_o_to_device();
void host_arr_RGB_training_to_device();
void host_arr_ll_weight_f_to_device();
void host_arr_ll_weight_s_to_device();
void host_arr_ll_bias_f_to_device();
void host_arr_ll_bias_s_to_device();
__global__ void device_arr_RGB_assemble();
__global__ void device_arr_weight_f_assemble();
__global__ void device_arr_weight_s_assemble();
__global__ void device_arr_RGB_weight_f_assemble();
__global__ void device_arr_RGB_weight_s_assemble();
__global__ void device_arr_bias_h_assemble();
__global__ void device_arr_bias_o_assemble();
__global__ void device_arr_RGB_bias_o_assemble();
__global__ void device_arr_RGB_training_assemble();
__global__ void device_arr_ll_weight_f_assemble();
__global__ void device_arr_ll_weight_s_assemble();
__global__ void device_arr_ll_bias_f_assemble();
__global__ void device_arr_ll_bias_s_assemble();
__global__ void device_arr_weight_f_demolish();
__global__ void device_arr_weight_s_demolish();
__global__ void device_arr_RGB_weight_f_demolish();
__global__ void device_arr_RGB_weight_s_demolish();
__global__ void device_arr_bias_h_demolish();
__global__ void device_arr_bias_o_demolish();
__global__ void device_arr_RGB_bias_o_demolish();
__global__ void device_arr_weight_f_demolish();
__global__ void device_arr_ll_weight_f_demolish();
__global__ void device_arr_ll_weight_s_demolish();
__global__ void device_arr_ll_bias_f_demolish();
__global__ void device_arr_ll_bias_s_demolish();
void device_arr_to_host();
void host_arr_weight_f_assemble();
void host_arr_weight_s_assemble();
void host_arr_RGB_weight_f_assemble();
void host_arr_RGB_weight_s_assemble();
void host_arr_bias_h_assemble();
void host_arr_bias_o_assemble();
void host_arr_RGB_bias_o_assemble();
void host_arr_ll_weight_f_assemble();
void host_arr_ll_weight_s_assemble();
void host_arr_ll_bias_f_assemble();
void host_arr_ll_bias_s_assemble();
__global__ void device_calc_sum_h(const int,const int,const int);
__global__ void device_add_bias_h(const int);
__global__ void device_calc_sum_o(const int ,const int,const int);
__global__ void device_add_bias_o(const int);
__global__ void device_ReLu_o(const int);
__global__ void device_sigmoid(const int);
__global__ void device_training_weight_s(const int,const float);//全体向けバックプロパゲーション
__global__ void device_training_bias_s(const int,const float);
__global__ void device_training_weight_f(const int,const float);
__global__ void device_training_bias_f(const int,const float,int,int);//
__global__ void device_show_results();
void forward_all(int);
__global__ void device_training_ws4(const float);//オートエンコーダ
__global__ void device_training_bias_s4(const float);
__global__ void device_training_wf4(const float);
__global__ void device_training_biasf4(const float);
__global__ void device_training_wf3(const float,int,int);
__global__ void device_training_biasf3(const float,int,int,int,int);
__global__ void device_training_wf2(const float,int,int,int,int);
__global__ void device_training_biasf2(const float,int,int,int,int,int,int);
__global__ void device_training_wf1(const float,int,int,int,int,int,int);
__global__ void device_training_biasf1(const float,int,int,int,int,int,int,int,int);
__global__ void device_training_wRGB(const float,int,int,int,int,int,int,int,int,int);
__global__ void device_training_bias0(const float,int,int,int,int,int,int,int,int,int);//
__global__ void device_training_ll_wf(const float);
__global__ void device_training_ll_ws(const float);
__global__ void device_training_ll_bf(const float);
__global__ void device_training_ll_bs(const float);
void back_prop(int,float,float,float,float,float,float,float);
__global__ void change_T_to_insects();
__global__ void change_T_to_leaves();
void autoencoder_I(int,int,int,float,float,float,float,float,float,float,float,float,float);
void autoencoder_L(int,int,int,float,float,float,float,float,float,float,float,float,float);
void save_specific_weights();
void load_specific_weights();
void save_general_weights();
void load_general_weights();
//__global__ void normalize();
__global__ void device_pass_values_to_ll_inputs();
__global__ void device_calc_ll_sum_h();
__global__ void device_ReLU_ll_h();
__global__ void device_calc_ll_sum_o();
__global__ void device_softmax_ll_o();
__global__ void device_stock_output_data(int num);
__global__ void device_check_data_INSECTS();
__global__ void device_check_data_LEAVES();
__global__ void Check_percentage_of_correct_answers();
//-------------------------------------------------------------------------------global_variable
const int pic_num(1000);  //画像枚数 
const int pic_size(32);  //ピクセルサイズ
const int level_of_NN(5);//NNの深さ(中間層の数)
const int hx_max(32),hy_max(32);//----------------特徴抽出層の中間層
const int hx_1(25),hy_1(25);
const int hx_2(20),hy_2(20);
const int hx_3(15),hy_3(15);
const int hx_4(12),hy_4(12);//--------------------------------------
const int tnum(10);//学習"させたい"回数(オートエンコーダ)
const int tnumBP(1);//学習"させたい"回数(バックプロパゲーション)
const int dnum(10);//学習"させたい"画像ディレクトリ数
const int pnum(1000);//学習"させたい"画像数
//const int hx_5(10),hy_5(10);//ラベルは中間層だが、別のNNの入力層として扱う
//const int hx_6(7),hy_6(7);
const int ox(1),oy(2);
//const int level_of_sum_h(level_of_hidden_layer);  //隠れ層への合計深さ
//const int shx_max(hx_max),shy_max(hx_max);
//const int level_of_sum_o(level_of_hidden_layer);  //出力層への合計深さ
//const int sox_max(pic_size),soy_max(pic_size);
//const int level_of_output_layer(level_of_hidden_layer);  //出力層の深さ
//const int ox_max(pic_size),oy_max(pic_size);
//const int level_of_error_layer(level_of_hidden_layer);  //誤差深さ
//const int ex_max(pic_size),ey_max(pic_size);
//const int level_of_weight_f(level_of_hidden_layer);  //重みf(first)深さ
//const int wx_f_target_max(hx_max),wy_f_target_max(hy_max),wx_f_root_max(pic_size),wy_f_root_max(pic_size);
//const int level_of_weight_s(level_of_hidden_layer);  //重みs(second)深さ
//const int wx_s_target_max(hx_max),wy_s_target_max(hy_max),wx_s_root_max(pic_size),wy_s_root_max(pic_size);
//const int level_of_bias_f(level_of_hidden_layer);  //バイアスf(first)深さ
//const int bx_f_max(hx_max),by_f_max(hy_max);
//const int level_of_bias_s(level_of_hidden_layer);  //バイアスs(second)深さ
//const int bx_s_max(pic_size),by_s_max(pic_size);
//const float a(0.0000000070),b(0.0000170);//中間層30*30
//const float a(0.0000000000060),b(0.000000060);//中間層50*50
//const float a(0.00000000090),b(0.0000000090);//hidden layer 20*20
//const float a(0.000000090),b(0.00000090);//hidden layer 40*40
//const float a(0.00000000090),b(0.0000000090);//hidden layer 5*5
float biggest_value_of_weight(0),smallest_value_of_weight(0);
float data_scale(0);
dim3 pic_num_pic_size(pic_num,pic_size,1);
dim3 level_of_NN_hx_max_hy_max(level_of_NN,hx_max,hy_max);
dim3 level_of_NN_pic_size_pic_size(level_of_NN,pic_size,pic_size);
dim3 hx_max_hy_max_pic_size(hx_max,hy_max,pic_size);
dim3 pic_size_pic_size_hx_max(pic_size,pic_size,hx_max);
dim3 level_of_NN_hx_max(level_of_NN,hx_max,1);
dim3 level_of_NN_pic_size(level_of_NN,pic_size,1);
dim3 hx_max_hy_max_hx_1(hx_max,hy_max,hx_1);
dim3 hx_1_hy_1_hx_2(hx_1,hy_1,hx_2);
dim3 hx_2_hy_2_hx_3(hx_2,hy_2,hx_3);
dim3 hx_3_hy_3_hx_4(hx_3,hy_3,hx_4);
dim3 hx_4_hy_4_hx_3(hx_4,hy_4,hx_3);
dim3 hx_4_hy_4(hx_4,hy_4,1);
dim3 hx_3_hy_3(hx_3,hy_3,1);
dim3 hx_3_hy_3_hx_2(hx_3,hy_3,hx_2);
dim3 hx_2_hy_2(hx_2,hy_2,1);
dim3 hx_2_hy_2_hx_1(hx_2,hy_2,hx_1);
dim3 hx_1_hy_1(hx_1,hy_1,1);
dim3 hx_1_hy_1_ox(hx_1,hy_1,ox);
dim3 hx_1_hy_1_hx_max(hx_1,hy_1,hx_max);
dim3 hx_max_hy_max(hx_max,hy_max,1);
dim3 hx_4_hy_4_hx_4(hx_4,hy_4,hx_4);
dim3 ox_oy_hx_4(ox,oy,hx_4);
dim3 hx_4_hy_4_ox(hx_4,hy_4,ox);
//-------------------------------------------------------------------------------
float host_arr_Blue[pic_num][pic_size][pic_size];
float host_arr_Green[pic_num][pic_size][pic_size];
float host_arr_Red[pic_num][pic_size][pic_size];
float host_arr_weight_f[level_of_NN][hx_max][hy_max][pic_size][pic_size];
float host_arr_weight_s[level_of_NN][pic_size][pic_size][hx_max][hy_max];
float host_arr_R_weight_f[hx_max][hy_max][pic_size][pic_size];
float host_arr_G_weight_f[hx_max][hy_max][pic_size][pic_size];
float host_arr_B_weight_f[hx_max][hy_max][pic_size][pic_size];
float host_arr_R_weight_s[pic_size][pic_size][hx_max][hy_max];
float host_arr_G_weight_s[pic_size][pic_size][hx_max][hy_max];
float host_arr_B_weight_s[pic_size][pic_size][hx_max][hy_max];
float host_arr_sum_h[level_of_NN][hx_max][hy_max];
float host_arr_sum_o[level_of_NN][pic_size][pic_size];
float host_arr_R_sum_o[pic_size][pic_size];
float host_arr_G_sum_o[pic_size][pic_size];
float host_arr_B_sum_o[pic_size][pic_size];
float host_arr_hidden[level_of_NN][hx_max][hy_max];
float host_arr_output[level_of_NN][pic_size][pic_size];
float host_arr_R_output[pic_size][pic_size];
float host_arr_G_output[pic_size][pic_size];
float host_arr_B_output[pic_size][pic_size];
float host_arr_bias_h[level_of_NN][hx_max][hy_max];
float host_arr_bias_o[level_of_NN][pic_size][pic_size];
float host_arr_R_bias_o[pic_size][pic_size];
float host_arr_G_bias_o[pic_size][pic_size];
float host_arr_B_bias_o[pic_size][pic_size];
float host_arr_R_trainer[pic_num][pic_size][pic_size];
float host_arr_G_trainer[pic_num][pic_size][pic_size];
float host_arr_B_trainer[pic_num][pic_size][pic_size];
float host_arr_last_layer_input[hx_4][hy_4];//最終層判定NN
float host_arr_last_layer_hidden[hx_4][hy_4];
float host_arr_last_layer_output[ox][oy];
float host_arr_last_layer_wf[hx_4][hy_4][hx_4][hy_4];
float host_arr_last_layer_ws[ox][oy][hx_4][hy_4];
float host_arr_last_layer_bf[hx_4][hy_4];
float host_arr_last_layer_bs[ox][oy];
__device__ float device_arr_Blue[pic_num][pic_size][pic_size];
__device__ float device_arr_Green[pic_num][pic_size][pic_size];
__device__ float device_arr_Red[pic_num][pic_size][pic_size];
__device__ float device_arr_weight_f[level_of_NN][hx_max][hy_max][pic_size][pic_size];
__device__ float device_arr_weight_s[level_of_NN][pic_size][pic_size][hx_max][hy_max];
__device__ float device_arr_R_weight_f[hx_max][hy_max][pic_size][pic_size];
__device__ float device_arr_G_weight_f[hx_max][hy_max][pic_size][pic_size];
__device__ float device_arr_B_weight_f[hx_max][hy_max][pic_size][pic_size];
__device__ float device_arr_R_weight_s[pic_size][pic_size][hx_max][hy_max];
__device__ float device_arr_G_weight_s[pic_size][pic_size][hx_max][hy_max];
__device__ float device_arr_B_weight_s[pic_size][pic_size][hx_max][hy_max];
__device__ float device_arr_sum_h[level_of_NN][hx_max][hy_max];
__device__ float device_arr_sum_o[level_of_NN][pic_size][pic_size];
__device__ float device_arr_R_sum_o[pic_size][pic_size];
__device__ float device_arr_G_sum_o[pic_size][pic_size];
__device__ float device_arr_B_sum_o[pic_size][pic_size];
__device__ float device_arr_hidden[level_of_NN][hx_max][hy_max];
__device__ float device_arr_output[level_of_NN][pic_size][pic_size];
__device__ float device_arr_R_output[pic_size][pic_size];
__device__ float device_arr_G_output[pic_size][pic_size];
__device__ float device_arr_B_output[pic_size][pic_size];
__device__ float device_arr_bias_h[level_of_NN][hx_max][hy_max];
__device__ float device_arr_bias_o[level_of_NN][pic_size][pic_size];
__device__ float device_arr_R_bias_o[pic_size][pic_size];
__device__ float device_arr_G_bias_o[pic_size][pic_size];
__device__ float device_arr_B_bias_o[pic_size][pic_size];
__device__ float device_arr_R_trainer[pic_num][pic_size][pic_size];
__device__ float device_arr_G_trainer[pic_num][pic_size][pic_size];
__device__ float device_arr_B_trainer[pic_num][pic_size][pic_size];
__device__ float device_arr_normalized[level_of_NN][hx_max][hy_max];
__device__ float device_arr_last_layer_input[hx_4][hy_4];
__device__ float device_arr_last_layer_hidden[hx_4][hy_4];
__device__ float device_arr_last_layer_output[ox][oy];
__device__ float device_arr_last_layer_sum_hidden[hx_4][hy_4];
__device__ float device_arr_last_layer_sum_output[ox][oy];
__device__ float device_arr_last_layer_wf[hx_4][hy_4][hx_4][hy_4];
__device__ float device_arr_last_layer_ws[ox][oy][hx_4][hy_4];
__device__ float device_arr_last_layer_bf[hx_4][hy_4];
__device__ float device_arr_last_layer_bs[ox][oy];
float host_arr_Blue1D[pic_num*pic_size*pic_size];
float host_arr_Green1D[pic_num*pic_size*pic_size];
float host_arr_Red1D[pic_num*pic_size*pic_size];
float host_arr_weight_f1D[level_of_NN*hx_max*hy_max*pic_size*pic_size];
float host_arr_weight_s1D[level_of_NN*pic_size*pic_size*hx_max*hy_max];
float host_arr_R_weight_f1D[hx_max*hy_max*pic_size*pic_size];
float host_arr_G_weight_f1D[hx_max*hy_max*pic_size*pic_size];
float host_arr_B_weight_f1D[hx_max*hy_max*pic_size*pic_size];
float host_arr_R_weight_s1D[pic_size*pic_size*hx_max*hy_max];
float host_arr_G_weight_s1D[pic_size*pic_size*hx_max*hy_max];
float host_arr_B_weight_s1D[pic_size*pic_size*hx_max*hy_max];
float host_arr_bias_h1D[level_of_NN*hx_max*hy_max];
float host_arr_bias_o1D[level_of_NN*pic_size*pic_size];
float host_arr_R_bias_o1D[pic_size*pic_size];
float host_arr_G_bias_o1D[pic_size*pic_size];
float host_arr_B_bias_o1D[pic_size*pic_size];
float host_arr_R_trainer1D[pic_num*pic_size*pic_size];
float host_arr_G_trainer1D[pic_num*pic_size*pic_size];
float host_arr_B_trainer1D[pic_num*pic_size*pic_size];
float host_arr_last_layer_input1D[hx_4][hy_4];
float host_arr_last_layer_hidden1D[hx_4][hy_4];
float host_arr_last_layer_output1D[ox][oy];
float host_arr_last_layer_wf1D[hx_4*hy_4*hx_4*hy_4];
float host_arr_last_layer_ws1D[ox*oy*hx_4*hy_4];
float host_arr_last_layer_bf1D[hx_4*hy_4];
float host_arr_last_layer_bs1D[ox*oy];
__device__ float device_arr_Blue1D[pic_num*pic_size*pic_size];
__device__ float device_arr_Green1D[pic_num*pic_size*pic_size];
__device__ float device_arr_Red1D[pic_num*pic_size*pic_size];
__device__ float device_arr_weight_f1D[level_of_NN*hx_max*hy_max*pic_size*pic_size];
__device__ float device_arr_weight_s1D[level_of_NN*pic_size*pic_size*hx_max*hy_max];
__device__ float device_arr_R_weight_f1D[hx_max*hy_max*pic_size*pic_size];
__device__ float device_arr_G_weight_f1D[hx_max*hy_max*pic_size*pic_size];
__device__ float device_arr_B_weight_f1D[hx_max*hy_max*pic_size*pic_size];
__device__ float device_arr_R_weight_s1D[pic_size*pic_size*hx_max*hy_max];
__device__ float device_arr_G_weight_s1D[pic_size*pic_size*hx_max*hy_max];
__device__ float device_arr_B_weight_s1D[pic_size*pic_size*hx_max*hy_max];
__device__ float device_arr_bias_h1D[level_of_NN*hx_max*hy_max];
__device__ float device_arr_bias_o1D[level_of_NN*pic_size*pic_size];
__device__ float device_arr_R_bias_o1D[pic_size*pic_size];
__device__ float device_arr_G_bias_o1D[pic_size*pic_size];
__device__ float device_arr_B_bias_o1D[pic_size*pic_size];
__device__ float device_arr_R_trainer1D[pic_num*pic_size*pic_size];
__device__ float device_arr_G_trainer1D[pic_num*pic_size*pic_size];
__device__ float device_arr_B_trainer1D[pic_num*pic_size*pic_size];
__device__ float device_arr_last_layer_input1D[hx_4*hy_4];
__device__ float device_arr_last_layer_hidden1D[hx_4*hy_4];
__device__ float device_arr_last_layer_output1D[ox*oy];
__device__ float device_arr_last_layer_wf1D[hx_4*hy_4*hx_4*hy_4];
__device__ float device_arr_last_layer_ws1D[ox*oy*hx_4*hy_4];
__device__ float device_arr_last_layer_bf1D[hx_4*hy_4];
__device__ float device_arr_last_layer_bs1D[ox*oy];
__device__ float device_arr_T[2];
__device__ float device_arr_output_stocked[dnum*pnum][ox][oy];
__device__ float device_arr_result_of_check[dnum*pnum];
//-------------------------------------------------------------------------------
//**************************************************************************************************************************************
int main(){
  //-------------------------------------------------------------------------------setting
  host_check_pic_insect();
  host_check_pic_leaves();
  //host_get_data_insect();
  //host_get_data_leaves();
  //host_arr_RGB_training_to_device();
  //device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();
  host_set_value_to_weight_f();
  host_set_value_to_weight_s();
  host_set_value_to_RGBweight_f();
  host_set_value_to_RGBweight_s();
  host_set_value_to_RGBbias_o();
  host_set_value_to_bias_h();
  host_set_value_to_bias_o();
  host_set_value_to_last_layer_wf();
  host_set_value_to_last_layer_ws();
  host_set_value_to_last_layer_bias_f();
  host_set_value_to_last_layer_bias_s();
  load_specific_weights();//学習済みの重みのロード
  load_general_weights();
  host_arr_weight_f_to_device();
  device_arr_weight_f_assemble<<<level_of_NN_hx_max_hy_max,pic_size>>>();
  host_arr_weight_s_to_device();
  device_arr_weight_s_assemble<<<level_of_NN_pic_size_pic_size,hx_max>>>();
  host_arr_RGB_weight_f_to_device();
  device_arr_RGB_weight_f_assemble<<<hx_max_hy_max_pic_size,pic_size>>>();
  host_arr_RGB_weight_s_to_device();
  device_arr_RGB_weight_s_assemble<<<pic_size_pic_size_hx_max,hy_max>>>();
  host_arr_bias_h_to_device();
  device_arr_bias_h_assemble<<<level_of_NN_hx_max,hy_max>>>();
  host_arr_bias_o_to_device();
  device_arr_bias_o_assemble<<<level_of_NN_pic_size,pic_size>>>();
  host_arr_RGB_bias_o_to_device();
  device_arr_RGB_bias_o_assemble<<<pic_size,pic_size>>>();
  host_arr_ll_weight_f_to_device();
  device_arr_ll_weight_f_assemble<<<hx_4_hy_4_hx_4,hy_4>>>();
  host_arr_ll_weight_s_to_device();
  device_arr_ll_weight_s_assemble<<<ox_oy_hx_4,hy_4>>>();
  host_arr_ll_bias_f_to_device();
  device_arr_ll_bias_f_assemble<<<hx_4,hy_4>>>();
  host_arr_ll_bias_s_to_device();
  device_arr_ll_bias_s_assemble<<<ox,oy>>>();
  //-------------------------------------------------------------------------------
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////process(NN)
  //Trainer配列を作成しオリジナル画像を渡す。また入力画像にノイズ付き画像を渡す。(フォルダ移動はループ追加で対処)
  ///*GPU
  //10-7-5-3-2
  //float Ca(0.000000020),Cb(0.000000020),Cc(0.00000000030),Cd(0.00000000030),Ce(0.0000000000030),Cf(0.0000000000030);//学習係数
  //float Cg(0.0000000000030),Ch(0.0000000000030),Ci(0.0000000000030),Cj(0.0000000000030);//学習係数
  //32-25-20-15-10
  float AEa(0.0000000050),AEb(0.0000000050),AEc(0.000000000000060),AEd(0.000000000000060),AEe(0.0000000000000000070),AEf(0.0000000000000000070);//AE学習係数
  float AEg(0.000000000000000000010),AEh(0.000000000000000000010),AEi(0.000000000000000000030),AEj(0.000000000000000000030);//AE学習係数
  float BPb(0.00000000000000000000030),BPd(0.00000000000000000000030),BPf(0.00000000000000000000030),BPh(0.00000000000000000000030),BPi(0.00000000000000000001);//BP学習係数
  float llalpha(0.30),llbeta(0.30);
  /*
  host_get_data_insect();
  host_arr_RGB_training_to_device();
  device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();
  autoencoder_I(tnum,pnum,dnum,AEa,AEb,AEc,AEd,AEe,AEf,AEg,AEh,AEi,AEj);
  
  host_get_data_leaves();
  host_arr_RGB_training_to_device();
  device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();
  autoencoder_L(tnum,pnum,dnum,AEa,AEb,AEc,AEd,AEe,AEf,AEg,AEh,AEi,AEj);
  */
  
  /*
  for(int cnt=0;cnt<3;cnt++){
    cout<<"<"<<cnt<<">"<<endl;
    host_get_data_insect();
    host_arr_RGB_training_to_device();
    device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();

    cout<<"Training Mode -> INSECTS"<<endl;
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      change_T_to_insects<<<1,1>>>();
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	device_clear_arr_ll<<<hx_4_hy_4_ox,oy>>>();
	forward_all(pictureNum);
	back_prop(pictureNum,BPi,BPh,BPf,BPd,BPb,llalpha,llbeta);
      }
    }
    device_show_results<<<1,1>>>();
    cudaDeviceSynchronize();
    
    host_get_data_leaves();
    host_arr_RGB_training_to_device();
    device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();
    cout<<"Training Mode -> LEAVES"<<endl;
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_leaves(directly_num);
      change_T_to_leaves<<<1,1>>>();
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	device_clear_arr_ll<<<hx_4_hy_4_ox,oy>>>();
	forward_all(pictureNum);
	back_prop(pictureNum,BPi,BPh,BPf,BPd,BPb,llalpha,llbeta);
      }
    }
    device_show_results<<<1,1>>>();
    cudaDeviceSynchronize();
  }
  */
  ///*
  ////////////////////////////////////////////////////////////////////////////////////////////////////test
  for(int cnt=0;cnt<1;cnt++){
    cout<<"<"<<cnt<<">"<<endl;
    host_get_data_insect();
    host_arr_RGB_training_to_device();
    device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();

    cout<<"Testing Mode -> INSECTS"<<endl;
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      change_T_to_insects<<<1,1>>>();
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	device_clear_arr_ll<<<hx_4_hy_4_ox,oy>>>();
	forward_all(pictureNum);
        device_show_results<<<1,1>>>();
	device_stock_output_data<<<1,1>>>(directly_num*pnum+pictureNum);
	//back_prop(pictureNum,BPi,BPh,BPf,BPd,BPb,llalpha,llbeta);
      }
    }
    cudaDeviceSynchronize();

    device_check_data_INSECTS<<<pnum*dnum,1>>>();
    Check_percentage_of_correct_answers<<<1,1>>>();
    
    host_get_data_leaves();
    host_arr_RGB_training_to_device();
    device_arr_RGB_training_assemble<<<pic_num_pic_size,pic_size>>>();
    cout<<"Testing Mode -> LEAVES"<<endl;
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_leaves(directly_num);
      change_T_to_leaves<<<1,1>>>();
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	device_clear_arr_ll<<<hx_4_hy_4_ox,oy>>>();
	forward_all(pictureNum);
        device_show_results<<<1,1>>>();
        device_stock_output_data<<<1,1>>>(directly_num*pnum+pictureNum);
	//back_prop(pictureNum,BPi,BPh,BPf,BPd,BPb,llalpha,llbeta);
      }
    }
    cudaDeviceSynchronize();

    device_check_data_LEAVES<<<dnum*pnum,1>>>();
    Check_percentage_of_correct_answers<<<1,1>>>();
  }
  ////////////////////////////////////////////////////////////////////////////////////////////////////test
  //*/
  device_arr_weight_f_demolish<<<level_of_NN_hx_max_hy_max,pic_size>>>();
  device_arr_weight_s_demolish<<<level_of_NN_pic_size_pic_size,hx_max>>>();
  device_arr_RGB_weight_f_demolish<<<hx_max_hy_max_pic_size,pic_size>>>();
  device_arr_RGB_weight_s_demolish<<<pic_size_pic_size_hx_max,hy_max>>>();
  device_arr_bias_h_demolish<<<level_of_NN_hx_max,hy_max>>>();
  device_arr_bias_o_demolish<<<level_of_NN_pic_size,pic_size>>>();
  device_arr_RGB_bias_o_demolish<<<pic_size,pic_size>>>();
  device_arr_ll_weight_f_demolish<<<hx_4_hy_4_hx_4,hy_4>>>();
  device_arr_ll_weight_s_demolish<<<ox_oy_hx_4,hy_4>>>();
  device_arr_ll_bias_f_demolish<<<hx_4,hy_4>>>();
  device_arr_ll_bias_s_demolish<<<ox,oy>>>();
  device_arr_to_host();
  host_arr_weight_f_assemble();
  host_arr_weight_s_assemble();
  host_arr_RGB_weight_f_assemble();
  host_arr_RGB_weight_s_assemble();
  host_arr_bias_h_assemble();
  host_arr_bias_o_assemble();
  host_arr_RGB_bias_o_assemble();
  host_arr_ll_weight_f_assemble();
  host_arr_ll_weight_s_assemble();
  host_arr_ll_bias_f_assemble();
  host_arr_ll_bias_s_assemble();
  //*/
  ///*CPU
  for(int train_num=0;train_num<1;train_num++){
    //cout<<train_num<<endl;
    for(int pictureNum=0;pictureNum<1;pictureNum++){
      clear_arr(0);//0はレベル
      host_calc_sum_RGBh(pictureNum);
      ReLu_h(0,hx_max,hy_max);//0はlevel
      host_calc_sum_RGBo();
      ReLu_RGBo();
      //training_RGBweight_s(pictureNum,0.00000000090);//0はpic_num
      //training_RGBbias_s(pictureNum,0.00000000090);
      //training_RGBweight_f(pictureNum,0.00000000090);
      //training_RGBbias_f(pictureNum,0.00000000090);
      //show_results(pictureNum);
      //device_show_results<<<hx_max,hy_max>>>();
      //device_show_results<<<ox,oy>>>();
      //device_show_results<<<hx_4,hy_4>>>();
    }
  }
  cudaDeviceSynchronize();
  get_scaleRGB();
  //save_specific_weights();//学習済みの重みの保存
  save_general_weights();//学習済みの重みの保存
  save_weights_with_image();
  save_output_result_with_image(0);//指定された番号で出力画像を保存
  //*/
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  return 0;
}
//**************************************************************************************************************************************
void device_ArrErrChk(cudaError_t adr){
  if(adr!=cudaSuccess){cout<<"cuda kernel error: "<<adr<<endl;}
}
void host_check_pic_insect(){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    if(img.cols!=pic_size||img.rows!=pic_size){
      cout<<"Warning! : Picture["<<num<<"] <Size is being no match>"<<endl;
    }
  }
}
void host_get_data_insect(){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    for(int width=0;width<pic_size;width++){//img.cols
      for(int height=0;height<pic_size;height++){//img.rows
        Vec3b intensity = img.at<Vec3b>(height,width);
	host_arr_B_trainer[num][width][height] = intensity.val[0]/255.0;
        host_arr_G_trainer[num][width][height] = intensity.val[1]/255.0;
        host_arr_R_trainer[num][width][height] = intensity.val[2]/255.0;
	//cout<<num<<","<<width<<","<<height<<"="<<Green[num][width][height]<<endl;
      }
    }
  }
}
void host_check_pic_leaves(){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100/samplele_100/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100/samplele_100/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    if(img.cols!=pic_size||img.rows!=pic_size){
      cout<<"Warning! : Picture["<<num<<"] <Size is being no match>"<<endl;
    }
  }
}
void host_get_data_leaves(){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100/samplele_100/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100/samplele_100/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    for(int width=0;width<pic_size;width++){//img.cols
      for(int height=0;height<pic_size;height++){//img.rows
        Vec3b intensity = img.at<Vec3b>(height,width);
    	host_arr_B_trainer[num][width][height] = intensity.val[0]/255.0;
        host_arr_G_trainer[num][width][height] = intensity.val[1]/255.0;
        host_arr_R_trainer[num][width][height] = intensity.val[2]/255.0;
	//cout<<num<<","<<width<<","<<height<<"="<<Green[num][width][height]<<endl;
      }
    }
  }
}
void host_get_noisy_data_insect(int dir_num){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100noisy"+to_string(dir_num)+"/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100noisy"+to_string(dir_num)+"/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    for(int width=0;width<pic_size;width++){//img.cols
      for(int height=0;height<pic_size;height++){//img.rows
        Vec3b intensity = img.at<Vec3b>(height,width);
	host_arr_Blue[num][width][height] = intensity.val[0]/255.0;
        host_arr_Green[num][width][height] = intensity.val[1]/255.0;
        host_arr_Red[num][width][height] = intensity.val[2]/255.0;
	//cout<<num<<","<<width<<","<<height<<"="<<Green[num][width][height]<<endl;
      }
    }
  }
}
void host_get_noisy_data_leaves(int dir_num){
  string picFileNameString;
  const char* picFileNameChar;
  for(int num=0;num<pic_num;num++){
    picFileNameString="/home/kai/Desktop/Programs/DNN/samples_100noisy"+to_string(dir_num)+"/samplele_100/sample"+to_string(num)+".jpg";
    //picFileNameString="/home/kai/デスクトップ/Programs/opencv_cuda/DNN/samples_100noisy"+to_string(dir_num)+"/samplele_100/sample"+to_string(num)+".jpg";
    picFileNameChar=picFileNameString.c_str();
    Mat img=imread(picFileNameChar,1);
    if(img.empty()){
      cout<<picFileNameChar<<endl;
      cout<<"Picture file was not found."<<endl;
      break;
    }
    for(int width=0;width<pic_size;width++){//img.cols
      for(int height=0;height<pic_size;height++){//img.rows
        Vec3b intensity = img.at<Vec3b>(height,width);
	host_arr_Blue[num][width][height] = intensity.val[0]/255.0;
        host_arr_Green[num][width][height] = intensity.val[1]/255.0;
        host_arr_Red[num][width][height] = intensity.val[2]/255.0;
	//cout<<num<<","<<width<<","<<height<<"="<<Green[num][width][height]<<endl;
      }
    }
  }
}
void host_set_value_to_weight_f(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int i=0;i<level_of_NN;i++){
    cout<<"Setting weight(f) values..."<<float(i)/float(level_of_NN)*100<<"%"<<endl;
    for(int j=0;j<hx_max;j++){
      for(int k=0;k<hy_max;k++){
	for(int l=0;l<pic_size;l++){
	  for(int m=0;m<pic_size;m++){
	    float result=dist(engine);
	    host_arr_weight_f[i][j][k][l][m]=result/100.0;
	  }
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_weight_s(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int i=0;i<level_of_NN;i++){
    cout<<"Setting weight(f) values..."<<float(i)/float(level_of_NN)*100<<"%"<<endl;
    for(int j=0;j<pic_size;j++){
      for(int k=0;k<pic_size;k++){
	for(int l=0;l<hx_max;l++){
	  for(int m=0;m<hy_max;m++){
	    float result=dist(engine);
	    host_arr_weight_s[i][j][k][l][m]=result/100.0;
	  }
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_RGBweight_f(){
  const int range(10);
  float result;
  for(int chg=0;chg<3;chg++){
    cout<<"Setting weight(RGBf) values..."<<float(chg)/float(3)*100<<"%"<<endl;
    random_device seed_gen;
    mt19937 engine(seed_gen());
    uniform_int_distribution<>dist(0,range);
    for(int i=0;i<hx_max;i++){
      for(int j=0;j<hx_max;j++){
	for(int k=0;k<pic_size;k++){
	  for(int l=0;l<pic_size;l++){
	    result=dist(engine);
	    if(chg==0)host_arr_R_weight_f[i][j][k][l]=result/100.0;
	    if(chg==1)host_arr_G_weight_f[i][j][k][l]=result/100.0;
	    if(chg==2)host_arr_B_weight_f[i][j][k][l]=result/100.0;
	    //cout<<host_arr_R_weight_f[i][j][k][l]<<endl;
	  }
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_RGBweight_s(){
  const int range(10);
  float result;
  for(int chg=0;chg<3;chg++){
    cout<<"Setting weight(RGBs) values..."<<float(chg)/float(3)*100<<"%"<<endl;
    random_device seed_gen;
    mt19937 engine(seed_gen());
    uniform_int_distribution<>dist(0,range);
    for(int i=0;i<pic_size;i++){
      for(int j=0;j<pic_size;j++){
	for(int k=0;k<hx_max;k++){
	  for(int l=0;l<hy_max;l++){
	    result=dist(engine);
	    if(chg==0)host_arr_R_weight_s[i][j][k][l]=result/100.0;
	    if(chg==1)host_arr_G_weight_s[i][j][k][l]=result/100.0;
	    if(chg==2)host_arr_B_weight_s[i][j][k][l]=result/100.0;
	    //cout<<host_arr_B_weight_s[i][j][k][l]<<endl;
	  }
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_last_layer_wf(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int i=0;i<hx_4;i++){
    cout<<"Setting last layer weight(f) values..."<<float(i)/float(hx_4)*100<<"%"<<endl;
    for(int j=0;j<hy_4;j++){
      for(int k=0;k<hx_4;k++){
	for(int l=0;l<hy_4;l++){
	  float result=dist(engine);
	  host_arr_last_layer_wf[i][j][k][l]=result/1000.0f;
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_last_layer_ws(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int i=0;i<ox;i++){
    cout<<"Setting last layer weight(s) values..."<<float(i)/float(ox)*100<<"%"<<endl;
    for(int j=0;j<oy;j++){
      for(int k=0;k<hx_4;k++){
	for(int l=0;l<hy_4;l++){
	  float result=dist(engine);
	  host_arr_last_layer_ws[i][j][k][l]=result/1000.0f;
	}
      }
    }
  }
  cout<<"done!"<<endl;
}
void host_set_value_to_bias_h(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	host_arr_bias_h[m][n][o]=1;
      }
    }
  }
}
void host_set_value_to_bias_o(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	host_arr_bias_o[m][n][o]=1;
      }
    }
  }
}
void host_set_value_to_RGBbias_o(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      host_arr_R_bias_o[m][n]=1;
      host_arr_G_bias_o[m][n]=1;
      host_arr_B_bias_o[m][n]=1;
    }
  }
}
void host_set_value_to_last_layer_bias_f(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      float result=dist(engine);
      host_arr_last_layer_bf[m][n]=result/10000.0f;
    }
  }
  //cout<<"done!"<<endl;
}
void host_set_value_to_last_layer_bias_s(){
  const int range(10);
  random_device seed_gen;
  mt19937 engine(seed_gen());
  uniform_int_distribution<>dist(0,range);
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      float result=dist(engine);
      host_arr_last_layer_bs[m][n]=result/10000.0f;
    }
  }
  //cout<<"done!"<<endl;
}
void host_calc_sum_RGBh(int pn){
  for(int m=0;m<hx_max;m++){
    for(int n=0;n<hy_max;n++){
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<pic_size;p++){
          host_arr_sum_h[0][m][n]=host_arr_sum_h[0][m][n]+host_arr_R_weight_f[m][n][o][p]*host_arr_Red[pn][o][p];
	  host_arr_sum_h[0][m][n]=host_arr_sum_h[0][m][n]+host_arr_G_weight_f[m][n][o][p]*host_arr_Green[pn][o][p];
	  host_arr_sum_h[0][m][n]=host_arr_sum_h[0][m][n]+host_arr_B_weight_f[m][n][o][p]*host_arr_Blue[pn][o][p];
	}
      }
      host_arr_sum_h[0][m][n]=host_arr_sum_h[0][m][n]+host_arr_bias_h[0][m][n];
      //cout<<host_arr_sum_h[0][m][n]<<endl;
      //cout<<host_arr_bias_h[0][m][n]<<endl;
    }
  }
}
__global__ void device_clear_arr(){
  int m=blockIdx.x;//hx_max
  int n=blockIdx.y;//hy_max
  int o=blockIdx.z;//pic_size
  int p=threadIdx.x;//pic_size
  for(int l=0;l<level_of_NN;l++){
    device_arr_sum_h[l][m][n]=0;
    device_arr_sum_o[l][o][p]=0;
    device_arr_R_sum_o[o][p]=0;
    device_arr_G_sum_o[o][p]=0;
    device_arr_B_sum_o[o][p]=0;
    //device_arr_hidden[l][m][n]=0;
    //device_arr_output[l][o][p]=0;
    device_arr_R_output[o][p]=0;
    device_arr_G_output[o][p]=0;
    device_arr_B_output[o][p]=0;
  }
}
__global__ void device_clear_arr_ll(){
  int m=blockIdx.x;//hx_4
  int n=blockIdx.y;//hy_4
  int o=blockIdx.z;//ox
  int p=threadIdx.x;//oy
  device_arr_last_layer_sum_hidden[m][n]=0;
  device_arr_last_layer_sum_output[o][p]=0;
}
__global__ void device_calc_sum_RGBh(int pn){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<pic_size;o++){
    for(int p=0;p<pic_size;p++){
      device_arr_sum_h[0][m][n]=device_arr_sum_h[0][m][n]+device_arr_R_weight_f[m][n][o][p]*device_arr_Red[pn][o][p];
      device_arr_sum_h[0][m][n]=device_arr_sum_h[0][m][n]+device_arr_G_weight_f[m][n][o][p]*device_arr_Green[pn][o][p];
      device_arr_sum_h[0][m][n]=device_arr_sum_h[0][m][n]+device_arr_B_weight_f[m][n][o][p]*device_arr_Blue[pn][o][p];
      //printf("%f\n",device_arr_sum_h[0][m][n]);
      //printf("%f\n",device_arr_B_weight_f[m][n][o][p]);
    }
  }
}
__global__ void device_add_bias_RGBh(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_sum_h[0][m][n]=device_arr_sum_h[0][m][n]+device_arr_bias_h[0][m][n];
}
void ReLu_h(int level,const int x_size,const int y_size){
  for(int m=0;m<x_size;m++){
    for(int n=0;n<y_size;n++){
      if(host_arr_sum_h[level][m][n]>=0){
        host_arr_hidden[level][m][n]=host_arr_sum_h[level][m][n];
	//cout<<host_arr_hidden[level][m][n]<<endl;
      }else{
	host_arr_hidden[level][m][n]=0;
      }
      //cout<<host_arr_hidden[level][m][n]<<endl;
    }
  }
}
__global__ void device_ReLu_h(const int level){
  int m=blockIdx.x;
  int n=threadIdx.x;
  if(device_arr_sum_h[level][m][n]>=0){
    device_arr_hidden[level][m][n]=device_arr_sum_h[level][m][n];
  }else{
    device_arr_hidden[level][m][n]=0;
  }
}
void host_calc_sum_RGBo(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<hx_max;o++){
	for(int p=0;p<hy_max;p++){
          host_arr_R_sum_o[m][n]=host_arr_R_sum_o[m][n]+host_arr_R_weight_s[m][n][o][p]*host_arr_hidden[0][o][p];
	  host_arr_G_sum_o[m][n]=host_arr_G_sum_o[m][n]+host_arr_G_weight_s[m][n][o][p]*host_arr_hidden[0][o][p];
	  host_arr_B_sum_o[m][n]=host_arr_B_sum_o[m][n]+host_arr_B_weight_s[m][n][o][p]*host_arr_hidden[0][o][p];
	}
      }
      host_arr_R_sum_o[m][n]=host_arr_R_sum_o[m][n]+host_arr_R_bias_o[m][n];
      host_arr_G_sum_o[m][n]=host_arr_G_sum_o[m][n]+host_arr_G_bias_o[m][n];
      host_arr_B_sum_o[m][n]=host_arr_B_sum_o[m][n]+host_arr_B_bias_o[m][n];
      //cout<<host_arr_R_sum_o[m][n]<<endl;
    }
  }
}
__global__ void device_calc_sum_RGBo(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<hx_max;o++){
    for(int p=0;p<hy_max;p++){
      device_arr_R_sum_o[m][n]=device_arr_R_sum_o[m][n]+device_arr_R_weight_s[m][n][o][p]*device_arr_hidden[0][o][p];
      device_arr_G_sum_o[m][n]=device_arr_G_sum_o[m][n]+device_arr_G_weight_s[m][n][o][p]*device_arr_hidden[0][o][p];
      device_arr_B_sum_o[m][n]=device_arr_B_sum_o[m][n]+device_arr_B_weight_s[m][n][o][p]*device_arr_hidden[0][o][p];
      //printf("%f\n",device_arr_R_sum_o[m][n]);
      //printf("%f\n",device_arr_B_weight_s[m][n][o][p]);
      //printf("%f\n",device_arr_hidden[0][o][p]);
    }
  }
}
__global__ void device_add_bias_RGBo(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_R_sum_o[m][n]=device_arr_R_sum_o[m][n]+device_arr_R_bias_o[m][n];
  device_arr_G_sum_o[m][n]=device_arr_G_sum_o[m][n]+device_arr_G_bias_o[m][n];
  device_arr_B_sum_o[m][n]=device_arr_B_sum_o[m][n]+device_arr_B_bias_o[m][n];
}
void ReLu_RGBo(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      if(host_arr_R_sum_o[m][n]>=0){
        host_arr_R_output[m][n]=host_arr_R_sum_o[m][n];
      }else{
	host_arr_R_output[m][n]=0;
      }
      if(host_arr_G_sum_o[m][n]>=0){
        host_arr_G_output[m][n]=host_arr_G_sum_o[m][n];
      }else{
	host_arr_G_output[m][n]=0;
      }
      if(host_arr_B_sum_o[m][n]>=0){
        host_arr_B_output[m][n]=host_arr_B_sum_o[m][n];
      }else{
	host_arr_B_output[m][n]=0;
      }
      //cout<<host_arr_R_output[m][n]<<endl;
      //cout<<host_arr_G_output[m][n]<<endl;
      //cout<<host_arr_B_output[m][n]<<endl;
    }
  }
}
__global__ void device_ReLu_RGBo(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  if(device_arr_R_sum_o[m][n]>=0){
    device_arr_R_output[m][n]=device_arr_R_sum_o[m][n];
  }else{
    device_arr_R_output[m][n]=0;
  }
  if(device_arr_G_sum_o[m][n]>=0){
    device_arr_G_output[m][n]=device_arr_G_sum_o[m][n];
  }else{
    device_arr_G_output[m][n]=0;
  }
  if(device_arr_B_sum_o[m][n]>=0){
    device_arr_B_output[m][n]=device_arr_B_sum_o[m][n];
  }else{
    device_arr_B_output[m][n]=0;
  }
  //printf("%f\n",device_arr_R_output[m][n]);
}
void training_RGBweight_s(int pn,const float a){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<hx_max;o++){
	for(int p=0;p<hy_max;p++){
	  /*
	  if(host_arr_R_weight_s[m][n][o][p]>1)host_arr_R_weight_s[m][n][o][p]=1;//制限
	  if(host_arr_R_weight_s[m][n][o][p]<0)host_arr_R_weight_s[m][n][o][p]=0;//制限
	  if(host_arr_G_weight_s[m][n][o][p]>1)host_arr_G_weight_s[m][n][o][p]=1;//制限
	  if(host_arr_G_weight_s[m][n][o][p]<0)host_arr_G_weight_s[m][n][o][p]=0;//制限
	  if(host_arr_B_weight_s[m][n][o][p]>1)host_arr_B_weight_s[m][n][o][p]=1;//制限
	  if(host_arr_B_weight_s[m][n][o][p]<0)host_arr_B_weight_s[m][n][o][p]=0;//制限
	  */
	  if(host_arr_R_sum_o[m][n]>=0){
	    host_arr_R_weight_s[m][n][o][p]=host_arr_R_weight_s[m][n][o][p]-a*host_arr_hidden[0][o][p]*(host_arr_R_output[m][n]-host_arr_R_trainer[pn][m][n]);
	    //cout<<a*host_arr_hidden[0][o][p]*(host_arr_R_output[m][n]-host_arr_Red[pn][m][n])<<endl;
	  }
	  if(host_arr_G_sum_o[m][n]>=0){
            host_arr_G_weight_s[m][n][o][p]=host_arr_G_weight_s[m][n][o][p]-a*host_arr_hidden[0][o][p]*(host_arr_G_output[m][n]-host_arr_G_trainer[pn][m][n]);
	    //cout<<host_arr_G_weight_s[m][n][o][p]<<endl;
	  }
	  if(host_arr_B_sum_o[m][n]>=0){
            host_arr_B_weight_s[m][n][o][p]=host_arr_B_weight_s[m][n][o][p]-a*host_arr_hidden[0][o][p]*(host_arr_B_output[m][n]-host_arr_B_trainer[pn][m][n]);
	    //cout<<host_arr_B_weight_s[m][n][o][p]<<endl;
	  }
	}
      }
    }
  }
}
__global__ void device_training_RGBweight_s(int pn,const float a){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  if(device_arr_R_sum_o[m][n]>=0){
    device_arr_R_weight_s[m][n][o][p]=device_arr_R_weight_s[m][n][o][p]-a*device_arr_hidden[0][o][p]*(device_arr_R_output[m][n]-device_arr_R_trainer[pn][m][n]);
    //cout<<a*device_arr_hidden[0][o][p]*(device_arr_R_output[m][n]-device_arr_Red[pn][m][n])<<endl;
  }
  if(device_arr_G_sum_o[m][n]>=0){
    device_arr_G_weight_s[m][n][o][p]=device_arr_G_weight_s[m][n][o][p]-a*device_arr_hidden[0][o][p]*(device_arr_G_output[m][n]-device_arr_G_trainer[pn][m][n]);
    //cout<<device_arr_G_weight_s[m][n][o][p]<<endl;
  }
  if(device_arr_B_sum_o[m][n]>=0){
    device_arr_B_weight_s[m][n][o][p]=device_arr_B_weight_s[m][n][o][p]-a*device_arr_hidden[0][o][p]*(device_arr_B_output[m][n]-device_arr_B_trainer[pn][m][n]);
    //cout<<device_arr_B_weight_s[m][n][o][p]<<endl;
  }
}
void training_RGBbias_s(int pn,const float a){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      if(host_arr_R_sum_o[m][n]>=0){
        host_arr_R_bias_o[m][n]=host_arr_R_bias_o[m][n]-a*(host_arr_R_output[m][n]-host_arr_R_trainer[pn][m][n]);
	//cout<<host_arr_R_bias_o[m][n]<<endl;
      }
      if(host_arr_G_sum_o[m][n]>=0){
        host_arr_G_bias_o[m][n]=host_arr_G_bias_o[m][n]-a*(host_arr_G_output[m][n]-host_arr_G_trainer[pn][m][n]);
	//cout<<host_arr_G_bias_o[m][n]<<endl;
      }
      if(host_arr_B_sum_o[m][n]>=0){
        host_arr_B_bias_o[m][n]=host_arr_B_bias_o[m][n]-a*(host_arr_B_output[m][n]-host_arr_B_trainer[pn][m][n]);
	//cout<<host_arr_B_bias_o[m][n]<<endl;
      }
    }
  }
}
__global__ void device_training_RGBbias_s(int pn,const float a){
  int m=blockIdx.x;
  int n=threadIdx.x;
  if(device_arr_R_sum_o[m][n]>=0){
    device_arr_R_bias_o[m][n]=device_arr_R_bias_o[m][n]-a*(device_arr_R_output[m][n]-device_arr_R_trainer[pn][m][n]);
    //cout<<device_arr_R_bias_o[m][n]<<endl;
  }
  if(device_arr_G_sum_o[m][n]>=0){
    device_arr_G_bias_o[m][n]=device_arr_G_bias_o[m][n]-a*(device_arr_G_output[m][n]-device_arr_G_trainer[pn][m][n]);
    //cout<<device_arr_G_bias_o[m][n]<<endl;
  }
  if(device_arr_B_sum_o[m][n]>=0){
    device_arr_B_bias_o[m][n]=device_arr_B_bias_o[m][n]-a*(device_arr_B_output[m][n]-device_arr_B_trainer[pn][m][n]);
    //cout<<device_arr_B_bias_o[m][n]<<endl;
  }
  //printf("%f\n",device_arr_B_bias_o[m][n]);
}
void training_RGBweight_f(int pn,const float b){
  float shorten_valueR(0);
  float shorten_valueG(0);
  float shorten_valueB(0);
  for(int m=0;m<hx_max;m++){
    for(int n=0;n<hy_max;n++){
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<pic_size;p++){
	  /*
	  if(host_arr_R_weight_f[m][n][o][p]>1)host_arr_R_weight_f[m][n][o][p]=1;//制限
	  if(host_arr_R_weight_f[m][n][o][p]<0)host_arr_R_weight_f[m][n][o][p]=0;//制限
	  if(host_arr_G_weight_f[m][n][o][p]>1)host_arr_G_weight_f[m][n][o][p]=1;//制限
	  if(host_arr_G_weight_f[m][n][o][p]<0)host_arr_G_weight_f[m][n][o][p]=0;//制限
	  if(host_arr_B_weight_f[m][n][o][p]>1)host_arr_B_weight_f[m][n][o][p]=1;//制限
	  if(host_arr_B_weight_f[m][n][o][p]<0)host_arr_B_weight_f[m][n][o][p]=0;//制限
	  */
	  shorten_valueR=host_arr_R_weight_s[o][p][m][n]*(host_arr_R_output[o][p]-host_arr_R_trainer[pn][o][p]);
          shorten_valueG=host_arr_G_weight_s[o][p][m][n]*(host_arr_G_output[o][p]-host_arr_G_trainer[pn][o][p]);
          shorten_valueB=host_arr_B_weight_s[o][p][m][n]*(host_arr_B_output[o][p]-host_arr_B_trainer[pn][o][p]);
	  if(host_arr_R_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
	    host_arr_R_weight_f[m][n][o][p]=host_arr_R_weight_f[m][n][o][p]-b*host_arr_Red[pn][o][p]*shorten_valueR;
	    host_arr_G_weight_f[m][n][o][p]=host_arr_G_weight_f[m][n][o][p]-b*host_arr_Green[pn][o][p]*shorten_valueR;
	    host_arr_B_weight_f[m][n][o][p]=host_arr_B_weight_f[m][n][o][p]-b*host_arr_Blue[pn][o][p]*shorten_valueR;
	    //cout<<a*host_arr_hidden[0][o][p]*(host_arr_R_output[m][n]-host_arr_Red[pn][m][n])<<endl;
	  }
	  if(host_arr_G_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
            host_arr_R_weight_f[m][n][o][p]=host_arr_R_weight_f[m][n][o][p]-b*host_arr_Red[pn][o][p]*shorten_valueG;
	    host_arr_G_weight_f[m][n][o][p]=host_arr_G_weight_f[m][n][o][p]-b*host_arr_Green[pn][o][p]*shorten_valueG;
	    host_arr_B_weight_f[m][n][o][p]=host_arr_B_weight_f[m][n][o][p]-b*host_arr_Blue[pn][o][p]*shorten_valueG;
	    //cout<<host_arr_G_weight_s[m][n][o][p]<<endl;
	  }
	  if(host_arr_B_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
            host_arr_R_weight_f[m][n][o][p]=host_arr_R_weight_f[m][n][o][p]-b*host_arr_Red[pn][o][p]*shorten_valueB;
	    host_arr_G_weight_f[m][n][o][p]=host_arr_G_weight_f[m][n][o][p]-b*host_arr_Green[pn][o][p]*shorten_valueB;
	    host_arr_B_weight_f[m][n][o][p]=host_arr_B_weight_f[m][n][o][p]-b*host_arr_Blue[pn][o][p]*shorten_valueB;
	    //cout<<host_arr_B_weight_s[m][n][o][p]<<endl;
	  }
	}
      }
    }
  }
}
__global__ void device_training_RGBweight_f(int pn,const float b){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  float shorten_valueR(0);
  float shorten_valueG(0);
  float shorten_valueB(0);
  if(device_arr_R_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
    shorten_valueR=device_arr_R_weight_s[o][p][m][n]*(device_arr_R_output[o][p]-device_arr_R_trainer[pn][o][p]);
    device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-b*device_arr_Red[pn][o][p]*shorten_valueR;
    device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-b*device_arr_Green[pn][o][p]*shorten_valueR;
    device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-b*device_arr_Blue[pn][o][p]*shorten_valueR;
    //cout<<a*device_arr_hidden[0][o][p]*(device_arr_R_output[m][n]-device_arr_Red[pn][m][n])<<endl;
  }
  if(device_arr_G_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
    shorten_valueG=device_arr_G_weight_s[o][p][m][n]*(device_arr_G_output[o][p]-device_arr_G_trainer[pn][o][p]);
    device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-b*device_arr_Red[pn][o][p]*shorten_valueG;
    device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-b*device_arr_Green[pn][o][p]*shorten_valueG;
    device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-b*device_arr_Blue[pn][o][p]*shorten_valueG;
    //cout<<device_arr_G_weight_s[m][n][o][p]<<endl;
  }
  if(device_arr_B_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
    shorten_valueB=device_arr_B_weight_s[o][p][m][n]*(device_arr_B_output[o][p]-device_arr_B_trainer[pn][o][p]);
    device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-b*device_arr_Red[pn][o][p]*shorten_valueB;
    device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-b*device_arr_Green[pn][o][p]*shorten_valueB;
    device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-b*device_arr_Blue[pn][o][p]*shorten_valueB;
    //cout<<device_arr_B_weight_s[m][n][o][p]<<endl;
  }
}
void training_RGBbias_f(int pn,const float b){
  float shorten_valueR(0);
  float shorten_valueG(0);
  float shorten_valueB(0);
  for(int m=0;m<hx_max;m++){
    for(int n=0;n<hy_max;n++){
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<pic_size;p++){
	  //if(host_arr_bias_h[0][m][n]>1)host_arr_bias_h[0][m][n]=1;//制限
	  //if(host_arr_bias_h[0][m][n]<0)host_arr_bias_h[0][m][n]=0;//制限
	  shorten_valueR=host_arr_R_weight_s[o][p][m][n]*(host_arr_R_output[o][p]-host_arr_R_trainer[pn][o][p]);
	  shorten_valueG=host_arr_G_weight_s[o][p][m][n]*(host_arr_G_output[o][p]-host_arr_G_trainer[pn][o][p]);
	  shorten_valueB=host_arr_B_weight_s[o][p][m][n]*(host_arr_B_output[o][p]-host_arr_B_trainer[pn][o][p]);
	  if(host_arr_R_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Red[pn][o][p]*/shorten_valueR;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Green[pn][o][p]*/shorten_valueR;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Blue[pn][o][p]*/shorten_valueR;
	    //cout<<a*host_arr_hidden[0][o][p]*(host_arr_R_output[m][n]-host_arr_Red[pn][m][n])<<endl;
	  }
	  if(host_arr_G_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
            host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Red[pn][o][p]*/shorten_valueG;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Green[pn][o][p]*/shorten_valueG;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Blue[pn][o][p]*/shorten_valueG;
	    //cout<<host_arr_G_weight_s[m][n][o][p]<<endl;
	  }
	  if(host_arr_B_sum_o[m][n]>=0 && host_arr_sum_h[0][m][n]>=0){
            host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Red[pn][o][p]*/shorten_valueB;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Green[pn][o][p]*/shorten_valueB;
	    host_arr_bias_h[0][m][n]=host_arr_bias_h[0][m][n]-b*/*host_arr_Blue[pn][o][p]*/shorten_valueB;
	    //cout<<host_arr_B_weight_s[m][n][o][p]<<endl;
	  }
	}
      }
    }
  }
}
__global__ void device_training_RGBbias_f(int pn,const float b){
  int m=blockIdx.x;
  int n=threadIdx.x;
  float shorten_valueR(0);
  float shorten_valueG(0);
  float shorten_valueB(0);
  for(int o=0;o<pic_size;o++){
    for(int p=0;p<pic_size;p++){
      if(device_arr_R_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
	shorten_valueR=device_arr_R_weight_s[o][p][m][n]*(device_arr_R_output[o][p]-device_arr_R_trainer[pn][o][p]);
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Red[pn][o][p]*/shorten_valueR;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Green[pn][o][p]*/shorten_valueR;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Blue[pn][o][p]*/shorten_valueR;
	//cout<<a*device_arr_hidden[0][o][p]*(device_arr_R_output[m][n]-device_arr_Red[pn][m][n])<<endl;
      }
      if(device_arr_G_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
	shorten_valueG=device_arr_G_weight_s[o][p][m][n]*(device_arr_G_output[o][p]-device_arr_G_trainer[pn][o][p]);
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Red[pn][o][p]*/shorten_valueG;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Green[pn][o][p]*/shorten_valueG;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Blue[pn][o][p]*/shorten_valueG;
	//cout<<device_arr_G_weight_s[m][n][o][p]<<endl;
      }
      if(device_arr_B_sum_o[m][n]>=0 && device_arr_sum_h[0][m][n]>=0){
	shorten_valueB=device_arr_B_weight_s[o][p][m][n]*(device_arr_B_output[o][p]-device_arr_B_trainer[pn][o][p]);
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Red[pn][o][p]*/shorten_valueB;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Green[pn][o][p]*/shorten_valueB;
	device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-b*/*device_arr_Blue[pn][o][p]*/shorten_valueB;
	//cout<<device_arr_B_weight_s[m][n][o][p]<<endl;
      }
    }
  }
  //printf("%f\n",device_arr_bias_h[0][m][n]);
}
void show_input_with_window(){
  Mat input(pic_size,pic_size,/*CV_32FC3*/CV_8UC3);
  int R(0),G(0),B(0);
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      B=255*host_arr_Blue[0][m][n];
      G=255*host_arr_Green[0][m][n];
      R=255*host_arr_Red[0][m][n];
      //cout<<int(255.0*host_arr_R_output[m][n])<<endl;
      input.at<Vec3b>(n,m) = Vec3b(B,G,R);
    }
  }
  imwrite("input.jpg",input);
}
void show_results(int pn){
  for(int m=0;m<hx_max/*pic_size*/;m++){
    for(int n=0;n<hy_max/*pic_size*/;n++){
      cout<<"output: "<<host_arr_R_output[m][n]<<" trainer: "<<host_arr_Red[pn][m][n]<<endl;
      cout<<"output: "<<host_arr_G_output[m][n]<<" trainer: "<<host_arr_Green[pn][m][n]<<endl;
      cout<<"output: "<<host_arr_B_output[m][n]<<" trainer: "<<host_arr_Blue[pn][m][n]<<endl;
    }
  }
}
void save_weights_with_image(){
  string saveImageName;
  const char* picFileNameChar;
  Mat weight(pic_size,pic_size,/*CV_32FC3*/CV_8UC3);
  int R(0),G(0),B(0);
  for(int m=0;m<10;m++){//hx_max最大数
    for(int n=0;n<10;n++){//hy_max最大数
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<pic_size;p++){
	  B=255*((smallest_value_of_weight+host_arr_B_weight_f[m][n][o][p])/biggest_value_of_weight);
	  G=255*((smallest_value_of_weight+host_arr_G_weight_f[m][n][o][p])/biggest_value_of_weight);
	  R=255*((smallest_value_of_weight+host_arr_R_weight_f[m][n][o][p])/biggest_value_of_weight);
	  weight.at<Vec3b>(o,p) = Vec3b(B,G,R);
	}
      }
      saveImageName="test"+to_string(hx_max*m+n)+".jpg";
      picFileNameChar=saveImageName.c_str();
      imwrite(picFileNameChar,weight);
    }
  }
}
void save_output_result_with_image(int num){
  string saveImageName;
  const char* picFileNameChar;
  Mat out(pic_size,pic_size,/*CV_32FC3*/CV_8UC3);
  int R(0),G(0),B(0);
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      B=255*host_arr_B_output[m][n];
      G=255*host_arr_G_output[m][n];
      R=255*host_arr_R_output[m][n];
      //cout<<int(255.0*host_arr_R_output[m][n])<<endl;
      out.at<Vec3b>(n,m) = Vec3b(B,G,R);
    }
  }
  saveImageName="out"+to_string(num)+".jpg";
  picFileNameChar=saveImageName.c_str();
  imwrite(picFileNameChar,out);
}
void clear_arr(int level){
  for(int m=0;m<hx_max;m++){
    for(int n=0;n<hy_max;n++){
      host_arr_sum_h[level][m][n]=0;
      host_arr_hidden[level][m][n]=0;
    }
  }
  for(int o=0;o<pic_size;o++){
    for(int p=0;p<pic_size;p++){
      host_arr_sum_o[level][o][p]=0;
      host_arr_R_sum_o[o][p]=0;
      host_arr_G_sum_o[o][p]=0;
      host_arr_B_sum_o[o][p]=0;
      host_arr_output[level][o][p]=0;
      host_arr_R_output[o][p]=0;
      host_arr_G_output[o][p]=0;
      host_arr_B_output[o][p]=0;
    }
  }
}
void get_scaleRGB(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      if(host_arr_R_weight_f[0][0][m][n] > biggest_value_of_weight)biggest_value_of_weight = host_arr_R_weight_f[0][0][m][n];
      if(host_arr_G_weight_f[0][0][m][n] > biggest_value_of_weight)biggest_value_of_weight = host_arr_G_weight_f[0][0][m][n];
      if(host_arr_B_weight_f[0][0][m][n] > biggest_value_of_weight)biggest_value_of_weight = host_arr_B_weight_f[0][0][m][n];
      if(host_arr_R_weight_f[0][0][m][n] < smallest_value_of_weight)smallest_value_of_weight = host_arr_R_weight_f[0][0][m][n];
      if(host_arr_G_weight_f[0][0][m][n] < smallest_value_of_weight)smallest_value_of_weight = host_arr_G_weight_f[0][0][m][n];
      if(host_arr_B_weight_f[0][0][m][n] < smallest_value_of_weight)smallest_value_of_weight = host_arr_B_weight_f[0][0][m][n];
    }
  }
  /*
  for(int o=0;o<hx_max;o++){
    for(int p=0;p<hy_max;p++){
      if(host_arr_R_weight_f[0][0][o][p]>biggest_value_of_weight)biggest_value_of_weight=host_arr_R_weight_f[0][0][o][p];
      if(host_arr_G_weight_f[0][0][o][p]>biggest_value_of_weight)biggest_value_of_weight=host_arr_G_weight_f[0][0][o][p];
      if(host_arr_B_weight_f[0][0][o][p]>biggest_value_of_weight)biggest_value_of_weight=host_arr_B_weight_f[0][0][o][p];
      if(host_arr_R_weight_f[0][0][o][p]<smallest_value_of_weight)smallest_value_of_weight=host_arr_R_weight_f[0][0][o][p];
      if(host_arr_G_weight_f[0][0][o][p]<smallest_value_of_weight)smallest_value_of_weight=host_arr_G_weight_f[0][0][o][p];
      if(host_arr_B_weight_f[0][0][o][p]<smallest_value_of_weight)smallest_value_of_weight=host_arr_B_weight_f[0][0][o][p];
    }
  }
  */
  //data_scale=abs(biggest_value_of_weight-smallest_value_of_weight);
}
void host_arr_RGB_to_device(){
  for(int m=0;m<pic_num;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	host_arr_Blue1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_Blue[m][n][o];
	host_arr_Green1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_Green[m][n][o];
	host_arr_Red1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_Red[m][n][o];
	//cout<<host_arr_Blue1D[m*pic_size*pic_size+n*pic_size+o]<<endl;
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_Blue1D,host_arr_Blue1D,pic_num*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_Green1D,host_arr_Green1D,pic_num*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_Red1D,host_arr_Red1D,pic_num*pic_size*pic_size*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_Blue1D,host_arr_Blue1D,pic_num*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_Green1D,host_arr_Green1D,pic_num*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_Red1D,host_arr_Red1D,pic_num*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_weight_f_to_device(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	for(int p=0;p<pic_size;p++){
	  for(int q=0;q<pic_size;q++){
	    host_arr_weight_f1D[m*hx_max*hy_max*pic_size*pic_size + n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=host_arr_weight_f[m][n][o][p][q];
	  }
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_weight_f1D,host_arr_weight_f1D,level_of_NN*hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_weight_f1D,host_arr_weight_f1D,level_of_NN*hx_max*hy_max*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_weight_s_to_device(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<hx_max;p++){
	  for(int q=0;q<hy_max;q++){
	    host_arr_weight_s1D[m*pic_size*pic_size*hx_max*hy_max + n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=host_arr_weight_s[m][n][o][p][q];
	  }
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_weight_s1D,host_arr_weight_s1D,level_of_NN*pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  //cudaMemcpy(device_arr_weight_s1D,host_arr_weight_s1D,level_of_NN*pic_size*pic_size*hx_max*hy_max*sizeof(float),cudaMemcpyHostToDevice);
}
void host_arr_RGB_weight_f_to_device(){
  for(int n=0;n<hx_max;n++){
    for(int o=0;o<hy_max;o++){
      for(int p=0;p<pic_size;p++){
	for(int q=0;q<pic_size;q++){
	  host_arr_R_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=host_arr_R_weight_f[n][o][p][q];
	  host_arr_G_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=host_arr_G_weight_f[n][o][p][q];
	  host_arr_B_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=host_arr_B_weight_f[n][o][p][q];
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_R_weight_f1D,host_arr_R_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_G_weight_f1D,host_arr_G_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_B_weight_f1D,host_arr_B_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_R_weight_f1D,host_arr_R_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_G_weight_f1D,host_arr_G_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_B_weight_f1D,host_arr_B_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_RGB_weight_s_to_device(){
  for(int n=0;n<pic_size;n++){
    for(int o=0;o<pic_size;o++){
      for(int p=0;p<hx_max;p++){
	for(int q=0;q<hy_max;q++){
	  host_arr_R_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=host_arr_R_weight_s[n][o][p][q];
	  host_arr_G_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=host_arr_G_weight_s[n][o][p][q];
	  host_arr_B_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=host_arr_B_weight_s[n][o][p][q];
	  //cout<<host_arr_R_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]<<endl;
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_R_weight_s1D,host_arr_R_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_G_weight_s1D,host_arr_G_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_B_weight_s1D,host_arr_B_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_R_weight_s1D,host_arr_R_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_G_weight_s1D,host_arr_G_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_B_weight_s1D,host_arr_B_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_bias_h_to_device(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	host_arr_bias_h1D[m*hx_max*hy_max+n*hy_max+o]=host_arr_bias_h[m][n][o];
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_bias_h1D,host_arr_bias_h1D,level_of_NN*hx_max*hy_max*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_bias_h1D,host_arr_bias_h1D,level_of_NN*hx_max*hy_max*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_bias_o_to_device(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	host_arr_bias_o1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_bias_o[m][n][o];
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_bias_o1D,host_arr_bias_o1D,level_of_NN*pic_size*pic_size*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_bias_o1D,host_arr_bias_o1D,level_of_NN*pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_RGB_bias_o_to_device(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      host_arr_R_bias_o1D[m*pic_size+n]=host_arr_R_bias_o[m][n];
      host_arr_G_bias_o1D[m*pic_size+n]=host_arr_G_bias_o[m][n];
      host_arr_B_bias_o1D[m*pic_size+n]=host_arr_B_bias_o[m][n];
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_R_bias_o1D,host_arr_R_bias_o1D,pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_G_bias_o1D,host_arr_G_bias_o1D,pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_B_bias_o1D,host_arr_B_bias_o1D,pic_size*pic_size*sizeof(float)));
  //device_ArrErrChk(cudaMemcpy(device_arr_R_bias_o1D,host_arr_R_bias_o1D,pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_G_bias_o1D,host_arr_G_bias_o1D,pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
  //device_ArrErrChk(cudaMemcpy(device_arr_B_bias_o1D,host_arr_B_bias_o1D,pic_size*pic_size*sizeof(float),cudaMemcpyHostToDevice));
}
void host_arr_RGB_training_to_device(){
  for(int m=0;m<pic_num;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	host_arr_B_trainer1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_B_trainer[m][n][o];
	host_arr_G_trainer1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_G_trainer[m][n][o];
	host_arr_R_trainer1D[m*pic_size*pic_size+n*pic_size+o]=host_arr_R_trainer[m][n][o];
	//cout<<host_arr_Blue1D[m*pic_size*pic_size+n*pic_size+o]<<endl;
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_B_trainer1D,host_arr_B_trainer1D,pic_num*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_G_trainer1D,host_arr_G_trainer1D,pic_num*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_R_trainer1D,host_arr_R_trainer1D,pic_num*pic_size*pic_size*sizeof(float)));
}
void host_arr_ll_weight_f_to_device(){
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  host_arr_last_layer_wf1D[m*hy_4*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p]=host_arr_last_layer_wf[m][n][o][p];
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_last_layer_wf1D,host_arr_last_layer_wf1D,hx_4*hy_4*hx_4*hy_4*sizeof(float)));
}
void host_arr_ll_weight_s_to_device(){
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  host_arr_last_layer_ws1D[m*oy*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p]=host_arr_last_layer_ws[m][n][o][p];
	}
      }
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_last_layer_ws1D,host_arr_last_layer_ws1D,ox*oy*hx_4*hy_4*sizeof(float)));
}
void host_arr_ll_bias_f_to_device(){
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      host_arr_last_layer_bf1D[m*hy_4+n]=host_arr_last_layer_bf[m][n];
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_last_layer_bf1D,host_arr_last_layer_bf1D,hx_4*hy_4*sizeof(float)));
}
void host_arr_ll_bias_s_to_device(){
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      host_arr_last_layer_bs1D[m*oy+n]=host_arr_last_layer_bs[m][n];
    }
  }
  device_ArrErrChk(cudaMemcpyToSymbol(device_arr_last_layer_bs1D,host_arr_last_layer_bs1D,ox*oy*sizeof(float)));
}
__global__ void device_arr_RGB_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_Red[m][n][o]=device_arr_Red1D[m*pic_size*pic_size+n*pic_size+o];
  device_arr_Green[m][n][o]=device_arr_Green1D[m*pic_size*pic_size+n*pic_size+o];
  device_arr_Blue[m][n][o]=device_arr_Blue1D[m*pic_size*pic_size+n*pic_size+o];
  //printf("%f\n",device_arr_Red1D[m*pic_size*pic_size+n*pic_size+o]);
  //printf("%f\n",device_arr_Red[m][n][o]);
}
__global__ void device_arr_weight_f_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  for(int q=0;q<pic_size;q++){
    device_arr_weight_f[m][n][o][p][q]=device_arr_weight_f1D[m*hx_max*hy_max*pic_size*pic_size + n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
    //printf("%f\n",device_arr_weight_f[m][n][o][p][q]);
  }
}
__global__ void device_arr_weight_s_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  for(int q=0;q<hy_max;q++){
    device_arr_weight_s[m][n][o][p][q]=device_arr_weight_s1D[m*pic_size*pic_size*hx_max*hy_max + n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
    //printf("%f\n",device_arr_weight_s[m][n][o][p][q]);
  }
}
__global__ void device_arr_RGB_weight_f_assemble(){
  int n=blockIdx.x;
  int o=blockIdx.y;
  int p=blockIdx.z;
  int q=threadIdx.x;
  device_arr_R_weight_f[n][o][p][q]=device_arr_R_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
  device_arr_G_weight_f[n][o][p][q]=device_arr_G_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
  device_arr_B_weight_f[n][o][p][q]=device_arr_B_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
  //printf("%f\n",device_arr_R_weight_f[n][o][p][q]);
}
__global__ void device_arr_RGB_weight_s_assemble(){
  int n=blockIdx.x;
  int o=blockIdx.y;
  int p=blockIdx.z;
  int q=threadIdx.x;
  device_arr_R_weight_s[n][o][p][q]=device_arr_R_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
  device_arr_G_weight_s[n][o][p][q]=device_arr_G_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
  device_arr_B_weight_s[n][o][p][q]=device_arr_B_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
  //printf("%f\n",device_arr_R_weight_s[n][o][p][q]);
  //printf("%f\n",device_arr_B_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]);
}
__global__ void device_arr_bias_h_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_bias_h[m][n][o]=device_arr_bias_h1D[m*hx_max*hy_max+n*hy_max+o];
  //printf("%f\n",device_arr_bias_h[m][n][o]);
}
__global__ void device_arr_bias_o_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_bias_o[m][n][o]=device_arr_bias_o1D[m*pic_size*pic_size+n*pic_size+o]; 
}
__global__ void device_arr_RGB_bias_o_assemble(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_R_bias_o[m][n]=device_arr_R_bias_o1D[m*pic_size+n];
  device_arr_G_bias_o[m][n]=device_arr_G_bias_o1D[m*pic_size+n];
  device_arr_B_bias_o[m][n]=device_arr_B_bias_o1D[m*pic_size+n]; 
}
__global__ void device_arr_RGB_training_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_R_trainer[m][n][o]=device_arr_R_trainer1D[m*pic_size*pic_size+n*pic_size+o];
  device_arr_G_trainer[m][n][o]=device_arr_G_trainer1D[m*pic_size*pic_size+n*pic_size+o];
  device_arr_B_trainer[m][n][o]=device_arr_B_trainer1D[m*pic_size*pic_size+n*pic_size+o];
  //printf("%f\n",device_arr_Red1D[m*pic_size*pic_size+n*pic_size+o]);
  //printf("%f\n",device_arr_Red[m][n][o]);
}
__global__ void device_arr_ll_weight_f_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  device_arr_last_layer_wf[m][n][o][p]=device_arr_last_layer_wf1D[m*hy_4*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p];
}
__global__ void device_arr_ll_weight_s_assemble(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  device_arr_last_layer_ws[m][n][o][p]=device_arr_last_layer_ws1D[m*oy*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p];
}
__global__ void device_arr_ll_bias_f_assemble(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_last_layer_bf[m][n]=device_arr_last_layer_bf1D[m*hy_4+n];
}
__global__ void device_arr_ll_bias_s_assemble(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_last_layer_bs[m][n]=device_arr_last_layer_bs1D[m*oy+n];
}
__global__ void device_arr_weight_f_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  for(int q=0;q<pic_size;q++){
    device_arr_weight_f1D[m*hx_max*hy_max*pic_size*pic_size + n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=device_arr_weight_f[m][n][o][p][q];
  }
}
__global__ void device_arr_weight_s_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  for(int q=0;q<hy_max;q++){
    device_arr_weight_s1D[m*pic_size*pic_size*hx_max*hy_max + n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=device_arr_weight_s[m][n][o][p][q];
  }
}
__global__ void device_arr_RGB_weight_f_demolish(){
  int n=blockIdx.x;
  int o=blockIdx.y;
  int p=blockIdx.z;
  int q=threadIdx.x;
  device_arr_R_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=device_arr_R_weight_f[n][o][p][q];
  device_arr_G_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=device_arr_G_weight_f[n][o][p][q];
  device_arr_B_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q]=device_arr_B_weight_f[n][o][p][q]; 
}
__global__ void device_arr_RGB_weight_s_demolish(){
  int n=blockIdx.x;
  int o=blockIdx.y;
  int p=blockIdx.z;
  int q=threadIdx.x;
  device_arr_R_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=device_arr_R_weight_s[n][o][p][q];
  device_arr_G_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=device_arr_G_weight_s[n][o][p][q];
  device_arr_B_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q]=device_arr_B_weight_s[n][o][p][q]; 
}
__global__ void device_arr_bias_h_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_bias_h1D[m*hx_max*hy_max+n*hy_max+o]=device_arr_bias_h[m][n][o];
  //printf("%f\n",device_arr_bias_h1D[m*hx_max*hy_max+n*hy_max+o]);
  //printf("%f\n",device_arr_bias_h[0][n][o]);
}
__global__ void device_arr_bias_o_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=threadIdx.x;
  device_arr_bias_o1D[m*pic_size*pic_size+n*pic_size+o]=device_arr_bias_o[m][n][o]; 
}
__global__ void device_arr_RGB_bias_o_demolish(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_R_bias_o1D[m*pic_size+n]=device_arr_R_bias_o[m][n];
  device_arr_G_bias_o1D[m*pic_size+n]=device_arr_G_bias_o[m][n];
  device_arr_B_bias_o1D[m*pic_size+n]=device_arr_B_bias_o[m][n]; 
}
__global__ void device_arr_ll_weight_f_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  device_arr_last_layer_wf1D[m*hy_4*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p]=device_arr_last_layer_wf[m][n][o][p];
}
__global__ void device_arr_ll_weight_s_demolish(){
  int m=blockIdx.x;
  int n=blockIdx.y;
  int o=blockIdx.z;
  int p=threadIdx.x;
  device_arr_last_layer_ws1D[m*oy*hx_4*hy_4+n*hx_4*hy_4+o*hy_4+p]=device_arr_last_layer_ws[m][n][o][p];
}
__global__ void device_arr_ll_bias_f_demolish(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_last_layer_bf1D[m*hy_4+n]=device_arr_last_layer_bf[m][n];
}
__global__ void device_arr_ll_bias_s_demolish(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_last_layer_bs1D[m*oy+n]=device_arr_last_layer_bs[m][n];
}
void device_arr_to_host(){
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_weight_f1D,device_arr_weight_f1D,level_of_NN*hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_weight_s1D,device_arr_weight_s1D,level_of_NN*pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_R_weight_f1D,device_arr_R_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_G_weight_f1D,device_arr_G_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_B_weight_f1D,device_arr_B_weight_f1D,hx_max*hy_max*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_R_weight_s1D,device_arr_R_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_G_weight_s1D,device_arr_G_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_B_weight_s1D,device_arr_B_weight_s1D,pic_size*pic_size*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_bias_h1D,device_arr_bias_h1D,level_of_NN*hx_max*hy_max*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_bias_o1D,device_arr_bias_o1D,level_of_NN*pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_R_bias_o1D,device_arr_R_bias_o1D,pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_G_bias_o1D,device_arr_G_bias_o1D,pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_B_bias_o1D,device_arr_B_bias_o1D,pic_size*pic_size*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_last_layer_wf1D,device_arr_last_layer_wf1D,hx_4*hy_4*hx_4*hy_4*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_last_layer_ws1D,device_arr_last_layer_ws1D,ox*oy*hx_4*hy_4*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_last_layer_bf1D,device_arr_last_layer_bf1D,hx_4*hy_4*sizeof(float)));
  device_ArrErrChk(cudaMemcpyFromSymbol(host_arr_last_layer_bs1D,device_arr_last_layer_bs1D,ox*oy*sizeof(float)));
}
void host_arr_weight_f_assemble(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	for(int p=0;p<pic_size;p++){
	  for(int q=0;q<pic_size;q++){
	    host_arr_weight_f[m][n][o][p][q]=host_arr_weight_f1D[m*hx_max*hy_max*pic_size*pic_size + n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
	    //cout<<<<endl;
	  }
	}
      }
    }
  }
}
void host_arr_weight_s_assemble(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	for(int p=0;p<hx_max;p++){
	  for(int q=0;q<hy_max;q++){
	    host_arr_weight_s[m][n][o][p][q]=host_arr_weight_s1D[m*pic_size*pic_size*hx_max*hy_max + n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
	  }
	}
      }
    }
  }
}
void host_arr_RGB_weight_f_assemble(){
  for(int n=0;n<hx_max;n++){
    for(int o=0;o<hy_max;o++){
      for(int p=0;p<pic_size;p++){
	for(int q=0;q<pic_size;q++){
	  host_arr_R_weight_f[n][o][p][q]=host_arr_R_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
	  host_arr_G_weight_f[n][o][p][q]=host_arr_G_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
	  host_arr_B_weight_f[n][o][p][q]=host_arr_B_weight_f1D[n*hy_max*pic_size*pic_size + o*pic_size*pic_size + p*pic_size + q];
	  //cout<<host_arr_R_weight_f[n][o][p][q]<<endl;
	}
      }
    }
  }
}
void host_arr_RGB_weight_s_assemble(){
  for(int n=0;n<pic_size;n++){
    for(int o=0;o<pic_size;o++){
      for(int p=0;p<hx_max;p++){
	for(int q=0;q<hy_max;q++){
	  host_arr_R_weight_s[n][o][p][q]=host_arr_R_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
	  host_arr_G_weight_s[n][o][p][q]=host_arr_G_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
	  host_arr_B_weight_s[n][o][p][q]=host_arr_B_weight_s1D[n*pic_size*hx_max*hy_max + o*hx_max*hy_max + p*hy_max + q];
	}
      }
    }
  }
}
void host_arr_bias_h_assemble(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<hx_max;n++){
      for(int o=0;o<hy_max;o++){
	host_arr_bias_h[m][n][o]=host_arr_bias_h1D[m*hx_max*hy_max+n*hy_max+o];
	//cout<<host_arr_bias_h[0][n][o]<<endl;
      }
    }
  }
}
void host_arr_bias_o_assemble(){
  for(int m=0;m<level_of_NN;m++){
    for(int n=0;n<pic_size;n++){
      for(int o=0;o<pic_size;o++){
	host_arr_bias_o[m][n][o]=host_arr_bias_o1D[m*pic_size*pic_size+n*pic_size+o];
      }
    }
  }
}
void host_arr_RGB_bias_o_assemble(){
  for(int m=0;m<pic_size;m++){
    for(int n=0;n<pic_size;n++){
      host_arr_R_bias_o[m][n]=host_arr_R_bias_o1D[m*pic_size+n];
      host_arr_G_bias_o[m][n]=host_arr_G_bias_o1D[m*pic_size+n];
      host_arr_B_bias_o[m][n]=host_arr_B_bias_o1D[m*pic_size+n];
      //cout<<host_arr_R_bias_o[m][n]<<endl;
    }
  }
}
/*
float host_arr_last_layer_wf1D[hx_4*hy_4*hx_4*hy_4];
float host_arr_last_layer_ws1D[ox*oy*hx_4*hy_4];
float host_arr_last_layer_bf1D[hx_4*hy_4];
float host_arr_last_layer_bs1D[ox*oy];
 */
void host_arr_ll_weight_f_assemble(){
  for(int n=0;n<hx_4;n++){
    for(int o=0;o<hy_4;o++){
      for(int p=0;p<hx_4;p++){
	for(int q=0;q<hy_4;q++){
          host_arr_last_layer_wf[n][o][p][q]=host_arr_last_layer_wf1D[n*hy_4*hx_4*hy_4+o*hx_4*hy_4+p*hy_4+q];
	}
      }
    }
  }
}
void host_arr_ll_weight_s_assemble(){
  for(int n=0;n<ox;n++){
    for(int o=0;o<oy;o++){
      for(int p=0;p<hx_4;p++){
	for(int q=0;q<hy_4;q++){
          host_arr_last_layer_ws[n][o][p][q]=host_arr_last_layer_ws1D[n*oy*hx_4*hy_4+o*hx_4*hy_4+p*hy_4+q];
	}
      }
    }
  }
}
void host_arr_ll_bias_f_assemble(){
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      host_arr_last_layer_bf[m][n]=host_arr_last_layer_bf1D[m*hy_4+n];
    }
  }
}
void host_arr_ll_bias_s_assemble(){
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      host_arr_last_layer_bs[m][n]=host_arr_last_layer_bs1D[m*oy+n];
    }
  }
}
__global__ void device_calc_sum_h(const int lv,const int ihx,const int ihy){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<ihx;o++){
    for(int p=0;p<ihy;p++){
      device_arr_sum_h[lv][m][n]=device_arr_sum_h[lv][m][n]+device_arr_weight_f[lv][m][n][o][p]*device_arr_sum_h[lv-1][o][p];
      device_arr_sum_h[lv][m][n]=device_arr_sum_h[lv][m][n]+device_arr_weight_f[lv][m][n][o][p]*device_arr_sum_h[lv-1][o][p];
      device_arr_sum_h[lv][m][n]=device_arr_sum_h[lv][m][n]+device_arr_weight_f[lv][m][n][o][p]*device_arr_sum_h[lv-1][o][p];
    }
  }
}
__global__ void device_add_bias_h(const int lv){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_sum_h[lv][m][n]=device_arr_sum_h[lv][m][n]+device_arr_bias_h[lv][m][n];
}
__global__ void device_calc_sum_o(const int lv,const int ihx,const int ihy){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<ihx;o++){
    for(int p=0;p<ihy;p++){
      device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_weight_s[lv][m][n][o][p]*device_arr_hidden[lv][o][p];
      device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_weight_s[lv][m][n][o][p]*device_arr_hidden[lv][o][p];
      device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_weight_s[lv][m][n][o][p]*device_arr_hidden[lv][o][p];
    }
  } 
}
__global__ void device_add_bias_o(const int lv){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_bias_o[lv][m][n];
  device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_bias_o[lv][m][n];
  device_arr_sum_o[lv][m][n]=device_arr_sum_o[lv][m][n]+device_arr_bias_o[lv][m][n];
}
__global__ void device_ReLu_o(const int lv){
  int m=blockIdx.x;
  int n=threadIdx.x;
  if(device_arr_sum_o[lv][m][n]>=0){
    device_arr_output[lv][m][n]=device_arr_sum_o[lv][m][n];
  }else{
    device_arr_output[lv][m][n]=0;
  }
  //printf("%f\n",device_arr_output[lv][m][n]);
}
__global__ void device_sigmoid(const int lv){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_output[lv][m][n]=1/(1+exp(-device_arr_sum_o[lv][m][n]));
}
__global__ void device_training_weight_s(const int lv,const float c){
  int m=blockIdx.x;//手前の中間層の座標X
  int n=blockIdx.y;//手前の中間層の座標Y
  int o=blockIdx.z;//現在の中間層の座標X
  int p=threadIdx.x;//現在の中間層の座標Y
  if(device_arr_sum_o[lv][m][n]>=0){
    device_arr_weight_s[lv][m][n][o][p]=device_arr_weight_s[lv][m][n][o][p]-c*device_arr_hidden[lv][o][p]*(device_arr_output[lv][m][n]-device_arr_hidden[lv-1][m][n]);
  }
}
__global__ void device_training_bias_s(const int lv,const float c){
  int m=blockIdx.x;//手前のの中間層の座標X
  int n=threadIdx.x;//手前の中間層の座標Y
  if(device_arr_sum_o[lv][m][n]>=0){
    device_arr_bias_o[lv][m][n]=device_arr_bias_o[lv][m][n]-c*(device_arr_output[lv][m][n]-device_arr_hidden[lv-1][m][n]);
  }
}
__global__ void device_training_weight_f(const int lv,const float d){
  int m=blockIdx.x;//手前の中間層の座標X
  int n=blockIdx.y;//手前の中間層の座標Y
  int o=blockIdx.z;//現在の中間層の座標X
  int p=threadIdx.x;//現在の中間層の座標Y
  float shorten_value(0);
  if(device_arr_sum_o[lv][m][n]>=0 && device_arr_sum_h[lv][o][p]>=0){
    shorten_value=device_arr_weight_s[lv][m][n][o][p]*(device_arr_output[lv][m][n]-device_arr_hidden[lv-1][m][n]);
    device_arr_weight_f[lv][o][p][m][n]=device_arr_weight_f[lv][o][p][m][n]-d*device_arr_hidden[lv-1][m][n]*shorten_value;
  }
}
__global__ void device_training_bias_f(const int lv,const float d,int m_size,int n_size){
  //int m=blockIdx.x;//手前の中間層の座標X
  //int n=blockIdx.y;//手前の中間層の座標Y
  int o=blockIdx.x;//現在の中間層の座標X
  int p=threadIdx.x;//現在の中間層の座標Y
  float shorten_value(0);
  for(int m=0;m<m_size;m++){
    for(int n=0;n<n_size;n++){
      if(device_arr_sum_o[lv][m][n]>=0 && device_arr_sum_h[lv][o][p]>=0){
        shorten_value=device_arr_weight_s[lv][m][n][o][p]*(device_arr_output[lv][m][n]-device_arr_hidden[lv-1][m][n]);
        device_arr_bias_h[lv][o][p]=device_arr_bias_h[lv][o][p]-d*shorten_value;
	//printf("%f\n",d*shorten_value);
      }
    }
  }
  //printf("%f\n",device_arr_bias_h[lv][o][p]);
}
__global__ void device_show_results(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  printf("output[0][0]: %lf(%lf)  Trainer(%lf)\n",device_arr_last_layer_output[0][0],device_arr_last_layer_sum_output[0][0],device_arr_T[0]);
  printf("output[0][1]: %lf(%lf)  Trainer(%lf)\n",device_arr_last_layer_output[0][1],device_arr_last_layer_sum_output[0][1],device_arr_T[1]);
  //printf("device_arr_output: %f  device_arr_trainer: %f\n",device_arr_R_output[m][n],device_arr_R_trainer[0][m][n]);
  //printf("device_arr_output: %f  device_arr_hidden: %f\n",device_arr_output[4][m][n],device_arr_hidden[3][m][n]);
  //printf("device_arr_output: %f  device_arr_hidden: %f\n",device_arr_output[4][m][n],device_arr_hidden[4][m][n]);
  //printf("device_arr_output: %f  device_arr_T: %f\n",device_arr_output[4][m][n],device_arr_T[n]);
  //printf("device_arr_sum_h: %f  device_arr_T: %f\n",device_arr_sum_h[3][m][n],device_arr_T);
}
void forward_all(int PN){
  device_calc_sum_RGBh<<<hx_max,hy_max>>>(PN);
  device_add_bias_RGBh<<<hx_max,hy_max>>>();
  device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
  device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);
  device_add_bias_h<<<hx_1,hy_1>>>(1);
  device_ReLu_h<<<hx_1,hy_1>>>(1);

  device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);
  device_add_bias_h<<<hx_2,hy_2>>>(2);
  device_ReLu_h<<<hx_2,hy_2>>>(2);

  device_calc_sum_h<<<hx_3,hy_3>>>(3,hx_2,hy_2);
  device_add_bias_h<<<hx_3,hy_3>>>(3);
  device_ReLu_h<<<hx_3,hy_3>>>(3);

  device_calc_sum_h<<<hx_4,hy_4>>>(4,hx_3,hy_3);
  device_add_bias_h<<<hx_4,hy_4>>>(4);
  device_ReLu_h<<<hx_4,hy_4>>>(4);

  device_pass_values_to_ll_inputs<<<hx_4,hy_4>>>();
  device_calc_ll_sum_h<<<hx_4,hy_4>>>();
  device_ReLU_ll_h<<<hx_4,hy_4>>>();
  device_calc_ll_sum_o<<<ox,oy>>>();
  device_softmax_ll_o<<<1,1>>>(); 
  
  /*
  device_calc_sum_o<<<1,2>>>(4,hx_4,hy_4);
  device_add_bias_o<<<1,2>>>(4);
  device_ReLu_o<<<1,2>>>(4);
  */
  //device_sigmoid<<<1,2>>>(4);
}
///*
__global__ void device_training_ws4(const float a){//バイアス系統は見直すべき
  int m=blockIdx.x;//hxs4
  int n=threadIdx.x;//hys4
  float shorten_value(0);
  //printf("ws4\n");
  /*sigmoid
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  device_arr_weight_s[4][0][0][m][n]=device_arr_weight_s[4][0][0][m][n]-a*device_arr_hidden[4][m][n]*shorten_value;//w5
  
  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);  
  device_arr_weight_s[4][0][1][m][n]=device_arr_weight_s[4][0][1][m][n]-a*device_arr_hidden[4][m][n]*shorten_value;//w5
  //printf("%f\n",device_arr_output[4][0][0]);
  */
  ///*ReLU
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  device_arr_weight_s[4][0][0][m][n]=device_arr_weight_s[4][0][0][m][n]-a*device_arr_hidden[4][m][n]*shorten_value;//w5
  
  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];  
  device_arr_weight_s[4][0][1][m][n]=device_arr_weight_s[4][0][1][m][n]-a*device_arr_hidden[4][m][n]*shorten_value;//w5
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //printf("%f\n",device_arr_output[4][0][0]);
  //*/
}
__global__ void device_training_bias_s4(const float a){
  //int m=blockIdx.x;
  //int n=threadIdx.x;
  float shorten_value(0);
  //printf("biass4\n");
  /*sigmid
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);  
  device_arr_bias_o[4][0][0]=device_arr_bias_o[4][0][0]-a*shorten_value;//bias5

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);  
  device_arr_bias_o[4][0][1]=device_arr_bias_o[4][0][1]-a*shorten_value;//bias5
  */
  ///*ReLu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];  
  device_arr_bias_o[4][0][0]=device_arr_bias_o[4][0][0]-a*shorten_value;//bias5

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];  
  device_arr_bias_o[4][0][1]=device_arr_bias_o[4][0][1]-a*shorten_value;//bias5
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //printf("%f\n",device_arr_bias_o[4][0][0]);
  //printf("%f\n",device_arr_bias_o[4][0][1]);
  //*/
}
__global__ void device_training_wf4(const float a){
  int m=blockIdx.x;//hxf4
  int n=blockIdx.y;//hyf4
  //int o=blockIdx.z;//hxf3
  int o=threadIdx.x;//hxf3
  int p=threadIdx.y;//hyf3
  float shorten_value(0);
  //printf("wf4\n");
  /*sigmoid
  if(device_arr_sum_h[4][m][n]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  device_arr_weight_f[4][m][n][o][p]=device_arr_weight_f[4][m][n][o][p]-a*device_arr_hidden[3][o][p]*device_arr_weight_s[4][0][0][m][n]*shorten_value;//w4

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  device_arr_weight_f[4][m][n][o][p]=device_arr_weight_f[4][m][n][o][p]-a*device_arr_hidden[3][o][p]*device_arr_weight_s[4][0][1][m][n]*shorten_value;//w4
  */
  ///*
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  device_arr_weight_f[4][m][n][o][p]=device_arr_weight_f[4][m][n][o][p]-a*device_arr_hidden[3][o][p]*device_arr_weight_s[4][0][0][m][n]*shorten_value;//w4

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  device_arr_weight_f[4][m][n][o][p]=device_arr_weight_f[4][m][n][o][p]-a*device_arr_hidden[3][o][p]*device_arr_weight_s[4][0][1][m][n]*shorten_value;//w4
  if(device_arr_sum_h[4][m][n]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_biasf4(const float a){
  int m=blockIdx.x;//hxf4
  int n=threadIdx.x;//hyf4
  float shorten_value(0);
  //printf("biasf4\n");
  /*sigmoid
  if(device_arr_sum_h[4][m][n]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);  
  device_arr_bias_h[4][m][n]=device_arr_bias_h[4][m][n]-a*device_arr_weight_s[4][0][0][m][n]*shorten_value;//bias4

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);  
  device_arr_bias_h[4][m][n]=device_arr_bias_h[4][m][n]-a*device_arr_weight_s[4][0][1][m][n]*shorten_value;//bias4
  */
  ///*ReLu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];  
  device_arr_bias_h[4][m][n]=device_arr_bias_h[4][m][n]-a*device_arr_weight_s[4][0][0][m][n]*shorten_value;//bias4

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];  
  device_arr_bias_h[4][m][n]=device_arr_bias_h[4][m][n]-a*device_arr_weight_s[4][0][1][m][n]*shorten_value;//bias4
  //printf("%f\n",device_arr_bias_h[4][m][n]);
  if(device_arr_sum_h[4][m][n]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_wf3(const float b,int hx4_size,int hy4_size){
  int m=blockIdx.x;//hxf3
  int n=blockIdx.y;//hyf3
  //int o=blockIdx.z;//hxf2
  int o=threadIdx.x;//hxf2
  int p=threadIdx.y;//hyf2
  float shorten_value(0),shorten_value2(0);
  //printf("wf3\n");
  /*sigmoid
  if(device_arr_sum_h[3][m][n]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_weight_f[3][m][n][o][p]=device_arr_weight_f[3][m][n][o][p]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//w3

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_weight_f[3][m][n][o][p]=device_arr_weight_f[3][m][n][o][p]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//w3
  */
  ///*ReLu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_weight_f[3][m][n][o][p]=device_arr_weight_f[3][m][n][o][p]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//w3

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_weight_f[3][m][n][o][p]=device_arr_weight_f[3][m][n][o][p]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//w3
  if(device_arr_sum_h[3][m][n]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_biasf3(const float b,int hx4_size,int hy4_size,int hx2_size,int hy2_size){//間違い
  int m=blockIdx.x;//hxf3
  int n=threadIdx.y;//hyf3
  //int o=blockIdx.z;//hxf2
  //int o=threadIdx.x;//hxf2
  //int p=threadIdx.y;//hyf2
  float shorten_value(0),shorten_value2(0);
  //printf("biasf3\n");
  /*sigmoid
  if(device_arr_sum_h[3][m][n]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_bias_h[3][m][n]=device_arr_bias_h[3][m][n]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//bias3

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_bias_h[3][m][n]=device_arr_bias_h[3][m][n]-b*device_arr_hidden[2][o][p]*shorten_value2*shorten_value;//bias3
  */
  ///*ReLu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_bias_h[3][m][n]=device_arr_bias_h[3][m][n]-b*device_arr_hidden[2][hx2_size][hy2_size]*shorten_value2*shorten_value;//bias3

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][m][n]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_bias_h[3][m][n]=device_arr_bias_h[3][m][n]-b*device_arr_hidden[2][hx2_size][hy2_size]*shorten_value2*shorten_value;//bias3
  if(device_arr_sum_h[3][m][n]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_wf2(const float c,int hx4_size,int hy4_size,int hx3_size,int hy3_size){//ここから修正
  int m=blockIdx.x;//hxf2
  int n=blockIdx.y;//hyf2
  //int o=blockIdx.z;//hxf1
  int o=threadIdx.x;//hxf1
  int p=threadIdx.y;//hyf1
  float shorten_value(0),shorten_value2(0);
  //printf("wf2\n");
  /*sigmoid
  if(device_arr_sum_h[2][m][n]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_weight_f[2][m][n][o][p]=device_arr_weight_f[2][m][n][o][p]-c*device_arr_hidden[1][o][p]*shorten_value2*shorten_value;//w2

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_weight_f[2][m][n][o][p]=device_arr_weight_f[2][m][n][o][p]-c*device_arr_hidden[1][o][p]*shorten_value2*shorten_value;//w2
  */
  ///*ReLu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_weight_f[2][m][n][o][p]=device_arr_weight_f[2][m][n][o][p]-c*device_arr_hidden[1][o][p]*shorten_value2*shorten_value;//w2

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_weight_f[2][m][n][o][p]=device_arr_weight_f[2][m][n][o][p]-c*device_arr_hidden[1][o][p]*shorten_value2*shorten_value;//w2
  if(device_arr_sum_h[2][m][n]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_biasf2(const float c,int hx4_size,int hy4_size,int hx3_size,int hy3_size,int hx1_size,int hy1_size){//間違い
  int m=blockIdx.x;//hxf2
  int n=threadIdx.y;//hyf2
  //int o=blockIdx.z;//hxf1
  //int o=threadIdx.x;//hxf1
  //int p=threadIdx.y;//hyf1
  float shorten_value(0),shorten_value2(0);
  //printf("biasf2\n");
  /*sigmoid
  if(device_arr_sum_h[2][m][n]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_bias_h[2][m][n]=device_arr_bias_h[2][m][n]-c*device_arr_weight_f[3][m][n][o][p]*shorten_value2*shorten_value;//bias2
    
  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_bias_h[2][m][n]=device_arr_bias_h[2][m][n]-c*device_arr_weight_f[3][m][n][o][p]*shorten_value2*shorten_value;//bias2
  */
  ///*Relu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  device_arr_bias_h[2][m][n]=device_arr_bias_h[2][m][n]-c*device_arr_weight_f[3][m][n][hx1_size][hy1_size]*shorten_value2*shorten_value;//bias2
    
  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][m][n]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_bias_h[2][m][n]=device_arr_bias_h[2][m][n]-c*device_arr_weight_f[3][m][n][hx1_size][hy1_size]*shorten_value2*shorten_value;//bias2
  if(device_arr_sum_h[2][m][n]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_wf1(const float d,int hx4_size,int hy4_size,int hx3_size,int hy3_size,int hx2_size,int hy2_size){
  int m=blockIdx.x;//hxf1
  int n=blockIdx.y;//hyf1
  //int o=blockIdx.z;//hxf0
  int o=threadIdx.x;//hxf0
  int p=threadIdx.y;//hyf0
  float shorten_value(0),shorten_value2(0),shorten_value3(0);
  //printf("wf1\n");
  /*sigmoid
  if(device_arr_sum_h[1][m][n]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  shorten_value3=device_arr_hidden[0][o][p]*device_arr_weight_f[2][hx2_size][hy2_size][m][n]*device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size];
  device_arr_weight_f[1][m][n][o][p]=device_arr_weight_f[1][m][n][o][p]-d*shorten_value3*shorten_value2*shorten_value;//w1

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  device_arr_weight_f[1][m][n][o][p]=device_arr_weight_f[1][m][n][o][p]-d*shorten_value3*shorten_value2*shorten_value;//w1
  */
  ///*Relu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  shorten_value3=device_arr_hidden[0][o][p]*device_arr_weight_f[2][hx2_size][hy2_size][m][n]*device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size];
  device_arr_weight_f[1][m][n][o][p]=device_arr_weight_f[1][m][n][o][p]-d*shorten_value3*shorten_value2*shorten_value;//w1

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_weight_f[1][m][n][o][p]=device_arr_weight_f[1][m][n][o][p]-d*shorten_value3*shorten_value2*shorten_value;//w1
  if(device_arr_sum_h[1][m][n]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_biasf1(const float d,int hx4_size,int hy4_size,int hx3_size,int hy3_size,int hx2_size,int hy2_size,int hx0_size,int hy0_size){//間違ってる
  int m=blockIdx.x;//hxf1
  int n=threadIdx.x;//hyf1
  //int o=blockIdx.z;//hxf0
  //int o=threadIdx.x;//hxf0
  //int p=threadIdx.y;//hyf0
  float shorten_value(0),shorten_value2(0),shorten_value3(0);
  //printf("biasf1\n");
  /*sigmoid
  if(device_arr_sum_h[1][m][n]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  shorten_value3=device_arr_hidden[0][o][p]*device_arr_weight_f[2][hx2_size][hy2_size][m][n]*device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size];
  device_arr_bias_h[1][m][n]=device_arr_bias_h[1][m][n]-d*device_arr_weight_f[2][m][n][o][p]*shorten_value3*shorten_value2*shorten_value;//bias1

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[[1]);
  device_arr_bias_h[1][m][n]=device_arr_bias_h[1][m][n]-d*device_arr_weight_f[2][m][n][o][p]*shorten_value3*shorten_value2*shorten_value;//bias1
  */
  ///*Relu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][0][hx4_size][hy4_size];
  shorten_value3=device_arr_hidden[0][hx0_size][hy0_size]*device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size];
  device_arr_bias_h[1][m][n]=device_arr_bias_h[1][m][n]-d*device_arr_weight_f[2][m][n][hx0_size][hy0_size]*shorten_value3*shorten_value2*shorten_value;//bias1

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  shorten_value2=device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size]*device_arr_weight_s[4][0][1][hx4_size][hy4_size];
  device_arr_bias_h[1][m][n]=device_arr_bias_h[1][m][n]-d*device_arr_weight_f[2][m][n][hx0_size][hy0_size]*shorten_value3*shorten_value2*shorten_value;//bias1
  if(device_arr_sum_h[1][m][n]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //*/
}
__global__ void device_training_wRGB(const float e,int pn,int hx4_size,int hy4_size,int hx3_size,int hy3_size,int hx2_size,int hy2_size,int hx1_size,int hy1_size){
  int m=blockIdx.x;//hxf0
  int n=blockIdx.y;//hyf0
  int o=blockIdx.z;//pic_size
  int p=threadIdx.x;//pic_size
  float shorten_value(0),shorten_value2(0),shorten_value3(0),shorten_value4(0),All(0);
  //printf("wRGB\n");
  /*sigmoid
  if(device_arr_sum_h[0][m][n]<0)return;
  if(device_arr_sum_h[1][hx1_size][hy1_size]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size];
  shorten_value3=device_arr_weight_f[1][hx1_size][hy1_size][m][n]*device_arr_weight_f[2][hx2_size][hy2_size][hx1_size][hy1_size];
  shorten_value4=device_arr_weight_s[4][hx4_size][hy4_size][hx3_size][hy3_size];
  device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-e*device_arr_Red[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R
  device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-e*device_arr_Green[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0G
  device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-e*device_arr_Blue[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0B

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-e*device_arr_Red[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R
  device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-e*device_arr_Green[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0G
  device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-e*device_arr_Blue[pn][o][p]*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0B
  */
  ///*Relu
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size];
  shorten_value3=device_arr_weight_f[1][hx1_size][hy1_size][m][n]*device_arr_weight_f[2][hx2_size][hy2_size][hx1_size][hy1_size];
  shorten_value4=device_arr_weight_s[4][hx4_size][hy4_size][hx3_size][hy3_size];
  All=shorten_value4*shorten_value3*shorten_value2;
  
  device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-e*device_arr_Red[pn][o][p]*All*shorten_value;//w0R
  device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-e*device_arr_Green[pn][o][p]*All*shorten_value;//w0G
  device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-e*device_arr_Blue[pn][o][p]*All*shorten_value;//w0B

  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  device_arr_R_weight_f[m][n][o][p]=device_arr_R_weight_f[m][n][o][p]-e*device_arr_Red[pn][o][p]*All*shorten_value;//w0R
  device_arr_G_weight_f[m][n][o][p]=device_arr_G_weight_f[m][n][o][p]-e*device_arr_Green[pn][o][p]*All*shorten_value;//w0G
  device_arr_B_weight_f[m][n][o][p]=device_arr_B_weight_f[m][n][o][p]-e*device_arr_Blue[pn][o][p]*All*shorten_value;//w0B
  //*/
  if(device_arr_sum_h[0][m][n]<0)return;
  if(device_arr_sum_h[1][hx1_size][hy1_size]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
}
__global__ void device_training_bias0(const float e,int pn,int hx4_size,int hy4_size,int hx3_size,int hy3_size,int hx2_size,int hy2_size,int hx1_size,int hy1_size){
  int m=blockIdx.x;//hxf0=hx_max
  int n=threadIdx.x;//hyf0=hy_max
  //int o=blockIdx.z;//hx1_size
  //int o=threadIdx.x;//hx1_size
  //int p=threadIdx.y;//hy1_size

  float shorten_value(0),shorten_value2(0),shorten_value3(0),shorten_value4(0);
  //printf("bias0\n");
  /*sigmoid
  if(device_arr_sum_h[0][m][n]<0)return;
  if(device_arr_sum_h[1][o][p]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  shorten_value=device_arr_output[4][0][0]*(1-device_arr_output[4][0][0])*(device_arr_output[4][0][0]-device_arr_T[0]);
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size];
  shorten_value3=device_arr_weight_f[1][o][p][m][n]*device_arr_weight_f[2][hx2_size][hy2_size][hx1_size][p];
  shorten_value4=device_arr_weight_s[4][hx4_size][hy4_size][hx3_size][hy3_size];
  device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-e*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R

  shorten_value=device_arr_output[4][0][1]*(1-device_arr_output[4][0][1])*(device_arr_output[4][0][1]-device_arr_T[1]);
  device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-e*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R
  */
  ///*Relu
  
  shorten_value=device_arr_output[4][0][0]-device_arr_T[0];
  shorten_value2=device_arr_weight_f[3][hx3_size][hy3_size][hx2_size][hy2_size]*device_arr_weight_f[4][hx4_size][hy4_size][hx3_size][hy3_size];
  shorten_value3=device_arr_weight_f[1][hx1_size][hy1_size][m][n]*device_arr_weight_f[2][hx2_size][hy2_size][hx1_size][hy1_size];
  shorten_value4=device_arr_weight_s[4][hx4_size][hy4_size][hx3_size][hy3_size];
  device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-e*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R
  
  shorten_value=device_arr_output[4][0][1]-device_arr_T[1];
  device_arr_bias_h[0][m][n]=device_arr_bias_h[0][m][n]-e*shorten_value4*shorten_value3*shorten_value2*shorten_value;//w0R
  if(device_arr_sum_h[0][m][n]<0)return;
  if(device_arr_sum_h[1][hx1_size][hy1_size]<0)return;
  if(device_arr_sum_h[2][hx2_size][hy2_size]<0)return;
  if(device_arr_sum_h[3][hx3_size][hy3_size]<0)return;
  if(device_arr_sum_h[4][hx4_size][hy4_size]<0)return;
  if(device_arr_sum_o[4][0][0]<0)return;
  if(device_arr_sum_o[4][0][1]<0)return;
  //printf("\n",);
}
__global__ void device_training_ll_wf(const float alpha){
  int m=blockIdx.x;//hx_4
  int n=blockIdx.y;//hy_4
  int o=blockIdx.z;//hx_4
  int p=threadIdx.x;//hy_4
  if(device_arr_last_layer_sum_hidden[m][n]<=0)return;
  float SV0((1-device_arr_last_layer_output[0][0])*(device_arr_last_layer_output[0][0]-device_arr_T[0]));
  device_arr_last_layer_wf[m][n][o][p]-=alpha*device_arr_last_layer_input[o][p]*device_arr_last_layer_ws[0][0][m][n]*device_arr_last_layer_output[0][0]*SV0;
  float SV1((1-device_arr_last_layer_output[0][1])*(device_arr_last_layer_output[0][1]-device_arr_T[1]));
  device_arr_last_layer_wf[m][n][o][p]-=alpha*device_arr_last_layer_input[o][p]*device_arr_last_layer_ws[0][1][m][n]*device_arr_last_layer_output[0][1]*SV1;
  //printf("%lf \n",device_arr_last_layer_wf[m][n][o][p]);
}
__global__ void device_training_ll_ws(const float alpha){
  int m=blockIdx.x;//ox
  int n=blockIdx.y;//oy
  int o=blockIdx.z;//hx_4
  int p=threadIdx.x;//hy_4
  float SV((1-device_arr_last_layer_output[m][n])*(device_arr_last_layer_output[m][n]-device_arr_T[n]));
  device_arr_last_layer_ws[m][n][o][p]-=alpha*device_arr_last_layer_hidden[o][p]*device_arr_last_layer_output[m][n]*SV;
}
__global__ void device_training_ll_bf(const float alpha){
  int m=blockIdx.x;//hx_4
  int n=threadIdx.x;//hy_4
  if(device_arr_last_layer_sum_hidden[m][n]<=0)return;
  float SV0((1-device_arr_last_layer_output[0][0])*(device_arr_last_layer_output[0][0]-device_arr_T[0]));
  device_arr_last_layer_bf[m][n]-=alpha*device_arr_last_layer_ws[0][0][m][n]*device_arr_last_layer_output[0][0]*SV0;
  float SV1((1-device_arr_last_layer_output[0][1])*(device_arr_last_layer_output[0][1]-device_arr_T[1]));
  device_arr_last_layer_bf[m][n]-=alpha*device_arr_last_layer_ws[0][1][m][n]*device_arr_last_layer_output[0][1]*SV1;
}
__global__ void device_training_ll_bs(const float alpha){
  float SV0((1-device_arr_last_layer_output[0][0])*(device_arr_last_layer_output[0][0]-device_arr_T[0]));
  device_arr_last_layer_bs[0][0]-=alpha*device_arr_last_layer_output[0][0]*SV0;
  float SV1((1-device_arr_last_layer_output[0][1])*(device_arr_last_layer_output[0][1]-device_arr_T[1]));
  device_arr_last_layer_bs[0][1]-=alpha*device_arr_last_layer_output[0][1]*SV1;
}
void back_prop(int pictureNum,float Ci,float Ch,float Cf,float Cd,float Cb,float lla,float llb){
  device_training_ll_ws<<<ox_oy_hx_4,hy_4>>>(lla);
  device_training_ll_bs<<<1,1>>>(lla);
  device_training_ll_wf<<<hx_4_hy_4_hx_4,hy_4>>>(llb);
  device_training_ll_bf<<<hx_4,hy_4>>>(llb);
  //device_training_ws4<<<hx_4,hy_4>>>(Ci);//Ci
  //device_training_bias_s4<<<1,1>>>(Ci);//Ci
  //device_training_wf4<<<hx_4_hy_4,hx_3_hy_3>>>(Ci);//Ci
  //device_training_biasf4<<<hx_4,hy_4>>>(Ci);//Ci
  /*
  for(int X1=0;X1<hx_4;X1++){
    for(int Y1=0;Y1<hy_4;Y1++){
      device_training_wf3<<<hx_3_hy_3,hx_2_hy_2>>>(Ch,X1,Y1);//Ch
      for(int X2=0;X2<hx_2;X2++){
        for(int Y2=0;Y2<hy_2;Y2++){
	  device_training_biasf3<<<hx_3,hy_3>>>(Ch,X1,Y1,X2,Y2);//Ch
        }
      }
    }
  }
  for(int X1=0;X1<hx_4;X1++){
    for(int Y1=0;Y1<hy_4;Y1++){
      for(int X2=0;X2<hx_3;X2++){
	for(int Y2=0;Y2<hy_3;Y2++){
	  device_training_wf2<<<hx_2_hy_2,hx_1_hy_1>>>(Cf,X1,Y1,X2,Y2);//Cf
	  for(int X3=0;X3<hx_1;X3++){
	    for(int Y3=0;Y3<hy_1;Y3++){
	      device_training_biasf2<<<hx_2,hy_2>>>(Cf,X1,Y1,X2,Y2,X3,Y3);//Cf
	    }
	  }
	}
      }
    }
  }
  for(int X1=0;X1<hx_4;X1++){
    for(int Y1=0;Y1<hy_4;Y1++){
      for(int X2=0;X2<hx_3;X2++){
	for(int Y2=0;Y2<hy_3;Y2++){
	  for(int X3=0;X3<hx_2;X3++){
	    for(int Y3=0;Y3<hy_2;Y3++){
	      device_training_wf1<<<hx_1_hy_1,hx_max_hy_max>>>(Cd,X1,Y1,X2,Y2,X3,Y3);//Cd
	      for(int X4=0;X4<hx_max;X4++){
		for(int Y4=0;Y4<hy_max;Y4++){
		  device_training_biasf1<<<hx_1,hy_1>>>(Cd,X1,Y1,X2,Y2,X3,Y3,X4,Y4);//Cd
		}
	      }
	    }
	  }
	}
      }
    }
  }
  for(int X1=0;X1<hx_4;X1++){
    for(int Y1=0;Y1<hy_4;Y1++){
      for(int X2=0;X2<hx_3;X2++){
	for(int Y2=0;Y2<hy_3;Y2++){
	  for(int X3=0;X3<hx_2;X3++){
	    for(int Y3=0;Y3<hy_2;Y3++){
	      for(int X4=0;X4<hx_1;X4++){
		for(int Y4=0;Y4<hy_1;Y4++){
		  device_training_wRGB<<<hx_max_hy_max_pic_size,pic_size>>>(Cb,pictureNum,X1,Y1,X2,Y2,X3,Y3,X4,Y4);//Cb
		  device_training_bias0<<<hx_max,hy_max>>>(Cb,pictureNum,X1,Y1,X2,Y2,X3,Y3,X4,Y4);//Cb
		}
	      }
	    }
	  }
	}
      }
    }
  }
  //*/
  
}
__global__ void change_T_to_insects(){
  device_arr_T[0]=1;
  device_arr_T[1]=0;
}
__global__ void change_T_to_leaves(){
  device_arr_T[0]=0;
  device_arr_T[1]=1;
}
void autoencoder_I(int tnum,int pnum,int dnum,float Ca,float Cb,float Cc,float Cd,float Ce,float Cf,float Cg,float Ch,float Ci,float Cj){
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ一層目]"<<endl;
    //device_show_results<<<pic_size,pic_size>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      //host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	device_calc_sum_RGBo<<<pic_size,pic_size>>>();
	device_add_bias_RGBo<<<pic_size,pic_size>>>();
	device_ReLu_RGBo<<<pic_size,pic_size>>>();
	device_training_RGBweight_s<<<pic_size_pic_size_hx_max,hy_max>>>(pictureNum,Ca);//学習(一層目)[オートエンコーダ]
	device_training_RGBbias_s<<<pic_size,pic_size>>>(pictureNum,Ca);
	device_training_RGBweight_f<<<hx_max_hy_max_pic_size,pic_size>>>(pictureNum,Cb);
	device_training_RGBbias_f<<<hx_max,hy_max>>>(pictureNum,Cb);
      }
    }
  }  
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ二層目]"<<endl;
    //device_show_results<<<hx_max,hy_max>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      //host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);
	device_calc_sum_o<<<hx_max,hy_max>>>(1,hx_1,hy_1);
	device_add_bias_o<<<hx_max,hy_max>>>(1);
	device_ReLu_o<<<hx_max,hy_max>>>(1);
	device_training_weight_s<<<hx_max_hy_max_hx_1,hy_1>>>(1,Cc);//学習(二層目)[オートエンコーダ]
	device_training_bias_s<<<hx_max,hy_max>>>(1,Cc);
	device_training_weight_f<<<hx_max_hy_max_hx_1,hy_1>>>(1,Cd);
	device_training_bias_f<<<hx_max,hy_max>>>(1,Cd,hx_1,hy_1);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ三層目]"<<endl;
    //device_show_results<<<hx_1,hy_1>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      //host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
        device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);
		
	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);
	device_calc_sum_o<<<hx_1,hy_1>>>(2,hx_2,hy_2);
	device_add_bias_o<<<hx_1,hy_1>>>(2);
	device_ReLu_o<<<hx_1,hy_1>>>(2);
	device_training_weight_s<<<hx_1_hy_1_hx_2,hy_2>>>(2,Ce);//学習(三層目)[オートエンコーダ]
	device_training_bias_s<<<hx_1,hy_1>>>(2,Ce);
	device_training_weight_f<<<hx_1_hy_1_hx_2,hy_2>>>(2,Cf);
	device_training_bias_f<<<hx_1,hy_1>>>(2,Cf,hx_2,hy_2);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ四層目]"<<endl;
    //device_show_results<<<hx_2,hy_2>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      //host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();

        device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel

	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);

	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);

	device_calc_sum_h<<<hx_3,hy_3>>>(3,hx_2,hy_2);//四層目[オートエンコーダ]
	device_add_bias_h<<<hx_3,hy_3>>>(3);
	device_ReLu_h<<<hx_3,hy_3>>>(3);
	device_calc_sum_o<<<hx_2,hy_2>>>(3,hx_3,hy_3);
	device_add_bias_o<<<hx_2,hy_2>>>(3);
	device_ReLu_o<<<hx_2,hy_2>>>(3);
	device_training_weight_s<<<hx_2_hy_2_hx_3,hy_3>>>(3,Cg);//学習(四層目)[オートエンコーダ]
	device_training_bias_s<<<hx_2,hy_2>>>(3,Cg);
	device_training_weight_f<<<hx_2_hy_2_hx_3,hy_3>>>(3,Ch);
	device_training_bias_f<<<hx_2,hy_2>>>(3,Ch,hx_3,hy_3);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ五層目]"<<endl;
    //device_show_results<<<hx_3,hy_3>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      host_get_noisy_data_insect(directly_num);
      //host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);

	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);

	device_calc_sum_h<<<hx_3,hy_3>>>(3,hx_2,hy_2);//四層目[オートエンコーダ]
	device_add_bias_h<<<hx_3,hy_3>>>(3);
	device_ReLu_h<<<hx_3,hy_3>>>(3);

	device_calc_sum_h<<<hx_4,hy_4>>>(4,hx_3,hy_3);//五層目[オートエンコーダ]
	device_add_bias_h<<<hx_4,hy_4>>>(4);
	device_ReLu_h<<<hx_4,hy_4>>>(4);
	device_calc_sum_o<<<hx_3,hy_3>>>(4,hx_4,hy_4);
	device_add_bias_o<<<hx_3,hy_3>>>(4);
	//device_sigmoid<<<hx_3,hy_3>>>(4);
	device_ReLu_o<<<hx_3,hy_3>>>(4);
	device_training_weight_s<<<hx_3_hy_3_hx_4,hy_4>>>(3,Ci);//学習(五層目)[オートエンコーダ]
	device_training_bias_s<<<hx_3,hy_3>>>(4,Ci);
	device_training_weight_f<<<hx_3_hy_3_hx_4,hy_4>>>(4,Cj);
	device_training_bias_f<<<hx_3,hy_3>>>(4,Cj,hx_4,hy_4);
      }
    }
  }
}
void autoencoder_L(int tnum,int pnum,int dnum,float Ca,float Cb,float Cc,float Cd,float Ce,float Cf,float Cg,float Ch,float Ci,float Cj){
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ一層目]"<<endl;
    //device_show_results<<<pic_size,pic_size>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      //host_get_noisy_data_insect(directly_num);
      host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	device_calc_sum_RGBo<<<pic_size,pic_size>>>();
	device_add_bias_RGBo<<<pic_size,pic_size>>>();
	device_ReLu_RGBo<<<pic_size,pic_size>>>();
	device_training_RGBweight_s<<<pic_size_pic_size_hx_max,hy_max>>>(pictureNum,Ca);//学習(一層目)[オートエンコーダ]
	device_training_RGBbias_s<<<pic_size,pic_size>>>(pictureNum,Ca);
	device_training_RGBweight_f<<<hx_max_hy_max_pic_size,pic_size>>>(pictureNum,Cb);
	device_training_RGBbias_f<<<hx_max,hy_max>>>(pictureNum,Cb);
      }
    }
  }  
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ二層目]"<<endl;
    //device_show_results<<<hx_max,hy_max>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      //host_get_noisy_data_insect(directly_num);
      host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);
	device_calc_sum_o<<<hx_max,hy_max>>>(1,hx_1,hy_1);
	device_add_bias_o<<<hx_max,hy_max>>>(1);
	device_ReLu_o<<<hx_max,hy_max>>>(1);
	device_training_weight_s<<<hx_max_hy_max_hx_1,hy_1>>>(1,Cc);//学習(二層目)[オートエンコーダ]
	device_training_bias_s<<<hx_max,hy_max>>>(1,Cc);
	device_training_weight_f<<<hx_max_hy_max_hx_1,hy_1>>>(1,Cd);
	device_training_bias_f<<<hx_max,hy_max>>>(1,Cd,hx_1,hy_1);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ三層目]"<<endl;
    //device_show_results<<<hx_1,hy_1>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      //host_get_noisy_data_insect(directly_num);
      host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
        device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);
		
	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);
	device_calc_sum_o<<<hx_1,hy_1>>>(2,hx_2,hy_2);
	device_add_bias_o<<<hx_1,hy_1>>>(2);
	device_ReLu_o<<<hx_1,hy_1>>>(2);
	device_training_weight_s<<<hx_1_hy_1_hx_2,hy_2>>>(2,Ce);//学習(三層目)[オートエンコーダ]
	device_training_bias_s<<<hx_1,hy_1>>>(2,Ce);
	device_training_weight_f<<<hx_1_hy_1_hx_2,hy_2>>>(2,Cf);
	device_training_bias_f<<<hx_1,hy_1>>>(2,Cf,hx_2,hy_2);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ四層目]"<<endl;
    //device_show_results<<<hx_2,hy_2>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      //host_get_noisy_data_insect(directly_num);
      host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();

        device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel

	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);

	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);

	device_calc_sum_h<<<hx_3,hy_3>>>(3,hx_2,hy_2);//四層目[オートエンコーダ]
	device_add_bias_h<<<hx_3,hy_3>>>(3);
	device_ReLu_h<<<hx_3,hy_3>>>(3);
	device_calc_sum_o<<<hx_2,hy_2>>>(3,hx_3,hy_3);
	device_add_bias_o<<<hx_2,hy_2>>>(3);
	device_ReLu_o<<<hx_2,hy_2>>>(3);
	device_training_weight_s<<<hx_2_hy_2_hx_3,hy_3>>>(3,Cg);//学習(四層目)[オートエンコーダ]
	device_training_bias_s<<<hx_2,hy_2>>>(3,Cg);
	device_training_weight_f<<<hx_2_hy_2_hx_3,hy_3>>>(3,Ch);
	device_training_bias_f<<<hx_2,hy_2>>>(3,Ch,hx_3,hy_3);
      }
    }
  }
  for(int train_num=0;train_num<tnum;train_num++){
    cout<<" 学習回数: "<<train_num<<" [オートエンコーダ五層目]"<<endl;
    //device_show_results<<<hx_3,hy_3>>>();
    for(int directly_num=0;directly_num<dnum;directly_num++){//用意したノイズ画像のフォルダの数だけ実行
      //host_get_noisy_data_insect(directly_num);
      host_get_noisy_data_leaves(directly_num);
      host_arr_RGB_to_device();
      device_arr_RGB_assemble<<<pic_num_pic_size,pic_size>>>();
      for(int pictureNum=0;pictureNum<pnum;pictureNum++){
	device_clear_arr<<<hx_max_hy_max_pic_size,pic_size>>>();
	
	device_calc_sum_RGBh<<<hx_max,hy_max>>>(pictureNum);//一層目[オートエンコーダ]
	device_add_bias_RGBh<<<hx_max,hy_max>>>();
	device_ReLu_h<<<hx_max,hy_max>>>(0);//0はlevel
	
	device_calc_sum_h<<<hx_1,hy_1>>>(1,hx_max,hy_max);//二層目[オートエンコーダ]
	device_add_bias_h<<<hx_1,hy_1>>>(1);
	device_ReLu_h<<<hx_1,hy_1>>>(1);

	device_calc_sum_h<<<hx_2,hy_2>>>(2,hx_1,hy_1);//三層目[オートエンコーダ]
	device_add_bias_h<<<hx_2,hy_2>>>(2);
	device_ReLu_h<<<hx_2,hy_2>>>(2);

	device_calc_sum_h<<<hx_3,hy_3>>>(3,hx_2,hy_2);//四層目[オートエンコーダ]
	device_add_bias_h<<<hx_3,hy_3>>>(3);
	device_ReLu_h<<<hx_3,hy_3>>>(3);

	device_calc_sum_h<<<hx_4,hy_4>>>(4,hx_3,hy_3);//五層目[オートエンコーダ]
	device_add_bias_h<<<hx_4,hy_4>>>(4);
	device_ReLu_h<<<hx_4,hy_4>>>(4);
	device_calc_sum_o<<<hx_3,hy_3>>>(4,hx_4,hy_4);
	device_add_bias_o<<<hx_3,hy_3>>>(4);
	//device_sigmoid<<<hx_3,hy_3>>>(4);
	device_ReLu_o<<<hx_3,hy_3>>>(4);
	device_training_weight_s<<<hx_3_hy_3_hx_4,hy_4>>>(3,Ci);//学習(五層目)[オートエンコーダ]
	device_training_bias_s<<<hx_3,hy_3>>>(4,Ci);
	device_training_weight_f<<<hx_3_hy_3_hx_4,hy_4>>>(4,Cj);
	device_training_bias_f<<<hx_3,hy_3>>>(4,Cj,hx_4,hy_4);
      }
    }
  }
}
void save_specific_weights(){
  //device_arr_R_weight_f[hx_max][hy_max][pic_size][pic_size]
  //device_arr_G_weight_f[hx_max][hy_max][pic_size][pic_size]
  //device_arr_B_weight_f[hx_max][hy_max][pic_size][pic_size]
  //device_arr_weight_f[level_of_NN][hx_max][hy_max][pic_size][pic_size]
  //device_arr_bias_h[level_of_NN][hx_max][hy_max]
  cout<<"Saving parameters..."<<endl;
  ofstream Rw;
  ofstream Gw;
  ofstream Bw;
  ofstream wf;
  ofstream bh;
  Rw.open("Rweights.txt",ios::trunc);
  Gw.open("Gweights.txt",ios::trunc);
  Bw.open("Bweights.txt",ios::trunc);
  wf.open("weightF.txt",ios::trunc);
  bh.open("biasH.txt",ios::trunc);
  
  for(int hx=0;hx<hx_max;hx++){
    for(int hy=0;hy<hy_max;hy++){
      for(int px=0;px<pic_size;px++){
	for(int py=0;py<pic_size;py++){
	  Rw<<host_arr_R_weight_f[hx][hy][px][py]<<endl;
	  Gw<<host_arr_G_weight_f[hx][hy][px][py]<<endl;
	  Bw<<host_arr_B_weight_f[hx][hy][px][py]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_1;hx_front++){
    for(int hy_front=0;hy_front<hy_1;hy_front++){
      for(int hx_back=0;hx_back<hx_max;hx_back++){
	for(int hy_back=0;hy_back<hy_max;hy_back++){
	  wf<<host_arr_weight_f[1][hx_front][hy_front][hx_back][hy_back]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_2;hx_front++){
    for(int hy_front=0;hy_front<hy_2;hy_front++){
      for(int hx_back=0;hx_back<hx_1;hx_back++){
	for(int hy_back=0;hy_back<hy_1;hy_back++){
	  wf<<host_arr_weight_f[2][hx_front][hy_front][hx_back][hy_back]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_3;hx_front++){
    for(int hy_front=0;hy_front<hy_3;hy_front++){
      for(int hx_back=0;hx_back<hx_2;hx_back++){
	for(int hy_back=0;hy_back<hy_2;hy_back++){
	  wf<<host_arr_weight_f[3][hx_front][hy_front][hx_back][hy_back]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_4;hx_front++){
    for(int hy_front=0;hy_front<hy_4;hy_front++){
      for(int hx_back=0;hx_back<hx_3;hx_back++){
	for(int hy_back=0;hy_back<hy_3;hy_back++){
	  wf<<host_arr_weight_f[4][hx_front][hy_front][hx_back][hy_back]<<endl;
	}
      }
    }
  }
  for(int hx=0;hx<hx_max;hx++){
    for(int hy=0;hy<hy_max;hy++){
      bh<<host_arr_bias_h[0][hx][hy]<<endl;
    }
  }
  for(int hx=0;hx<hx_1;hx++){
    for(int hy=0;hy<hy_1;hy++){
      bh<<host_arr_bias_h[1][hx][hy]<<endl;
    }
  }
  for(int hx=0;hx<hx_2;hx++){
    for(int hy=0;hy<hy_2;hy++){
      bh<<host_arr_bias_h[2][hx][hy]<<endl;
    }
  }
  for(int hx=0;hx<hx_3;hx++){
    for(int hy=0;hy<hy_3;hy++){
      bh<<host_arr_bias_h[3][hx][hy]<<endl;
    }
  }
  for(int hx=0;hx<hx_4;hx++){
    for(int hy=0;hy<hy_4;hy++){
      bh<<host_arr_bias_h[4][hx][hy]<<endl;
    }
  }
  Rw.close();
  Gw.close();
  Bw.close();
  wf.close();
  bh.close();
  cout<<"done!"<<endl;
}
void load_specific_weights(){
  cout<<"Loading parameters..."<<endl;
  ifstream Rw;
  ifstream Gw;
  ifstream Bw;
  ifstream wf;
  ifstream bh;
  Rw.open("Rweights.txt");
  Gw.open("Gweights.txt");
  Bw.open("Bweights.txt");
  wf.open("weightF.txt");
  bh.open("biasH.txt");
  
  for(int hx=0;hx<hx_max;hx++){
    for(int hy=0;hy<hy_max;hy++){
      for(int px=0;px<pic_size;px++){
	for(int py=0;py<pic_size;py++){
	  Rw>>host_arr_R_weight_f[hx][hy][px][py];
	  Gw>>host_arr_G_weight_f[hx][hy][px][py];
	  Bw>>host_arr_B_weight_f[hx][hy][px][py];
	  //cout<<host_arr_R_weight_f[hx][hy][px][py]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_1;hx_front++){
    for(int hy_front=0;hy_front<hy_1;hy_front++){
      for(int hx_back=0;hx_back<hx_max;hx_back++){
	for(int hy_back=0;hy_back<hy_max;hy_back++){
	  wf>>host_arr_weight_f[1][hx_front][hy_front][hx_back][hy_back];
	  //cout<<host_arr_weight_f[1][hx_front][hy_front][hx_back][hy_back]<<endl;
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_2;hx_front++){
    for(int hy_front=0;hy_front<hy_2;hy_front++){
      for(int hx_back=0;hx_back<hx_1;hx_back++){
	for(int hy_back=0;hy_back<hy_1;hy_back++){
	  wf>>host_arr_weight_f[2][hx_front][hy_front][hx_back][hy_back];
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_3;hx_front++){
    for(int hy_front=0;hy_front<hy_3;hy_front++){
      for(int hx_back=0;hx_back<hx_2;hx_back++){
	for(int hy_back=0;hy_back<hy_2;hy_back++){
	  wf>>host_arr_weight_f[3][hx_front][hy_front][hx_back][hy_back];
	}
      }
    }
  }
  for(int hx_front=0;hx_front<hx_4;hx_front++){
    for(int hy_front=0;hy_front<hy_4;hy_front++){
      for(int hx_back=0;hx_back<hx_3;hx_back++){
	for(int hy_back=0;hy_back<hy_3;hy_back++){
	  wf>>host_arr_weight_f[4][hx_front][hy_front][hx_back][hy_back];
	}
      }
    }
  }
  for(int hx=0;hx<hx_max;hx++){
    for(int hy=0;hy<hy_max;hy++){
      bh>>host_arr_bias_h[0][hx][hy];
    }
  }
  for(int hx=0;hx<hx_1;hx++){
    for(int hy=0;hy<hy_1;hy++){
      bh>>host_arr_bias_h[1][hx][hy];
    }
  }
  for(int hx=0;hx<hx_2;hx++){
    for(int hy=0;hy<hy_2;hy++){
      bh>>host_arr_bias_h[2][hx][hy];
    }
  }
  for(int hx=0;hx<hx_3;hx++){
    for(int hy=0;hy<hy_3;hy++){
      bh>>host_arr_bias_h[3][hx][hy];
    }
  }
  for(int hx=0;hx<hx_4;hx++){
    for(int hy=0;hy<hy_4;hy++){
      bh>>host_arr_bias_h[4][hx][hy];
    }
  }
  Rw.close();
  Gw.close();
  Bw.close();
  wf.close();
  bh.close();
  cout<<"done!";
}
void save_general_weights(){
  cout<<"Saving parameters..."<<endl;
  ofstream wf;
  ofstream ws;
  ofstream bf;
  ofstream bs;
  wf.open("weightsF(general).txt",ios::trunc);
  bf.open("biasF(general).txt",ios::trunc);
  ws.open("weightsS(general).txt",ios::trunc);
  bs.open("biasS(general).txt",ios::trunc);
  
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  wf<<host_arr_last_layer_wf[m][n][o][p]<<endl;
	}
      }
    }
  }
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  ws<<host_arr_last_layer_ws[m][n][o][p]<<endl;
	}
      }
    }
  }
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      bf<<host_arr_last_layer_bf[m][n]<<endl;
    }
  }
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      bs<<host_arr_last_layer_bs[m][n]<<endl;
    }
  }
  
  wf.close();
  bf.close();
  ws.close();
  bs.close();
  cout<<"done!"<<endl;
}
void load_general_weights(){
  ifstream wf;
  ifstream ws;
  ifstream bf;
  ifstream bs;
  wf.open("weightsF(general).txt");
  bf.open("biasF(general).txt");
  ws.open("weightsS(general).txt");
  bs.open("biasS(general).txt");
  
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  wf>>host_arr_last_layer_wf[m][n][o][p];
	}
      }
    }
  }
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      for(int o=0;o<hx_4;o++){
	for(int p=0;p<hy_4;p++){
	  ws>>host_arr_last_layer_ws[m][n][o][p];
	}
      }
    }
  }
  for(int m=0;m<hx_4;m++){
    for(int n=0;n<hy_4;n++){
      bf>>host_arr_last_layer_bf[m][n];
    }
  }
  for(int m=0;m<ox;m++){
    for(int n=0;n<oy;n++){
      bs>>host_arr_last_layer_bs[m][n];
    }
  }
  
  wf.close();
  bf.close();
  ws.close();
  bs.close();
  cout<<"done!"<<endl;
}
__global__ void device_pass_values_to_ll_inputs(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  device_arr_last_layer_input[m][n]=device_arr_hidden[4][m][n]/1000000000.0f;
  //printf("device_arr_last_layer_input: %lf\n",device_arr_last_layer_input[m][n]);
}
__global__ void device_calc_ll_sum_h(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<hx_4;o++){
    for(int p=0;p<hy_4;p++){
      device_arr_last_layer_sum_hidden[m][n]+=device_arr_last_layer_input[m][n]*device_arr_last_layer_wf[o][p][m][n]+device_arr_last_layer_bf[o][p];
    }
  }
  //printf("device_arr_last_layer_wf[o][p][m][n]: %lf\n",device_arr_last_layer_wf[0][0][0][0]);
}
__global__ void device_ReLU_ll_h(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  if(device_arr_last_layer_sum_hidden[m][n]<=0){
    device_arr_last_layer_hidden[m][n]=0;
    return;
  }
  device_arr_last_layer_hidden[m][n]=device_arr_last_layer_sum_hidden[m][n];
}
__global__ void device_calc_ll_sum_o(){
  int m=blockIdx.x;
  int n=threadIdx.x;
  for(int o=0;o<hx_4;o++){
    for(int p=0;p<hy_4;p++){
      device_arr_last_layer_sum_output[m][n]+=device_arr_last_layer_hidden[o][p]*device_arr_last_layer_ws[m][n][o][p]+device_arr_last_layer_bs[m][n];
    }
  }
  //printf("device_arr_last_layer_hidden[o][p]: %lf\n",device_arr_last_layer_hidden[0][0]);
}
__global__ void device_softmax_ll_o(){
  ///*
  float C(0);
  if(device_arr_last_layer_sum_output[0][0]>device_arr_last_layer_sum_output[0][1]){
    C=device_arr_last_layer_sum_output[0][0];
  }else{
    C=device_arr_last_layer_sum_output[0][1];
  }
  //*/
  float sum(exp(device_arr_last_layer_sum_output[0][0]-C)+exp(device_arr_last_layer_sum_output[0][1]-C));
  device_arr_last_layer_output[0][0]=exp(device_arr_last_layer_sum_output[0][0]-C)/sum;
  device_arr_last_layer_output[0][1]=exp(device_arr_last_layer_sum_output[0][1]-C)/sum;
  //device_arr_last_layer_output[0][0]=1/(1+exp(device_arr_last_layer_sum_output[0][0]));
  //device_arr_last_layer_output[0][1]=1/(1+exp(device_arr_last_layer_sum_output[0][1]));
  //printf("output[0][0]: %lf\n",device_arr_last_layer_output[0][0]);
  //printf("output[0][1]: %lf\n",device_arr_last_layer_output[0][1]);
  //printf("output[0][0]: %lf\n",device_arr_last_layer_sum_output[0][0]);
  //printf("output[0][1]: %lf\n",device_arr_last_layer_sum_output[0][1]);
}
__global__ void device_stock_output_data(int num){
  device_arr_output_stocked[num][0][0]=device_arr_last_layer_output[0][0];
  device_arr_output_stocked[num][0][1]=device_arr_last_layer_output[0][1];
  //printf("device_arr_output_stocked[num][0][0]=%lf\n",device_arr_output_stocked[num][0][0]);
  //printf("%lf>%lf\n",device_arr_output_stocked[num][0][0],device_arr_output_stocked[num][0][1]);
}
__global__ void device_check_data_INSECTS(){
  int num=blockIdx.x;
  if(device_arr_output_stocked[num][0][0]>device_arr_output_stocked[num][0][1]){
    device_arr_result_of_check[num]=1;
  }else{
    device_arr_result_of_check[num]=0;
  }
  //printf("%lf>%lf\n",device_arr_output_stocked[num][0][0],device_arr_output_stocked[num][0][1]);
}
__global__ void device_check_data_LEAVES(){
  int num=blockIdx.x;
  if(device_arr_output_stocked[num][0][0]<device_arr_output_stocked[num][0][1]){
    device_arr_result_of_check[num]=1;
  }else{
    device_arr_result_of_check[num]=0;
  }
  //printf("%lf>%lf\n",device_arr_output_stocked[num][0][0],device_arr_output_stocked[num][0][1]);
}
__global__ void Check_percentage_of_correct_answers(){
  int sum(0);
  for(int i=0;i<pnum*dnum;i++){
    sum+=device_arr_result_of_check[i];
  }
  printf("Percentage of correct answers: %lf\n",float(sum)/float(pnum*dnum));
}
