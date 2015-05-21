#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/util/densecrf/densecrf.h"
#include <cstdio>
#include <cmath>

namespace caffe {

// ======================================================================
// Phillip's NIPS 2011 : dense_inference.cpp
// ======================================================================
// Files to be added :
// header files : densecrf.h, fastmath.h , permutohedral.h , util.h , bipartitedensecrf.cpp , filter.cpp, sse_defs.h
// Src Files : densecrf.cpp , permutohedral.cpp , util.cpp

template <typename Dtype>
void DenseCRFLayer<Dtype>::DenseInference( unsigned char * im, float * unary, Dtype *res, int W, int H, int M) {
  
  // Setup the CRF model
  DenseCRF2D crf(W, H, M);
  // Specify the unary potential as an array of size W*H*(#classes)
  // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
  crf.setUnaryEnergy( unary );
  // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
  // x_stddev = 3
  // y_stddev = 3
  // weight = 3
  crf.addPairwiseGaussian( x_gauss_, y_gauss_, wt_gauss_ );
  // add a color dependent term (feature = xyrgb)
  // x_stddev = 60
  // y_stddev = 60
  // r_stddev = g_stddev = b_stddev = 20
  // weight = 10
  crf.addPairwiseBilateral( x_bilateral_, y_bilateral_, r_bilateral_, g_bilateral_, b_bilateral_, im, wt_bilateral_ );
  
  // Do map inference
  short * map = new short[W*H];
  crf.map(10, map);
  
  // Store the result. Memory already allocated to top.
  
  //LOG(INFO) << "\nDebugging Info for Output Score ";
  for( int k=0; k<W*H*1; k++ ){
    res[k] = (Dtype) map[k];
  //  if (k<=50)
  //    LOG(INFO)<<"Predicted Label(pixel "<<k<<" , label) : "<<res[k];
  }

  delete[] map;
}

// ======================================================================
// ======================================================================


template <typename Dtype>
void DenseCRFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  x_gauss_ = this->layer_param_.densecrf_param().x_gauss();
  y_gauss_ = this->layer_param_.densecrf_param().y_gauss();
  wt_gauss_ = this->layer_param_.densecrf_param().wt_gauss();

  x_bilateral_ = this->layer_param_.densecrf_param().x_bilateral();
  y_bilateral_ = this->layer_param_.densecrf_param().y_bilateral();
  r_bilateral_ = this->layer_param_.densecrf_param().r_bilateral();
  g_bilateral_ = this->layer_param_.densecrf_param().g_bilateral();
  b_bilateral_ = this->layer_param_.densecrf_param().b_bilateral();
  wt_bilateral_ = this->layer_param_.densecrf_param().wt_bilateral();

  LOG(INFO) << "Dense CRF Layer:" << std::endl
      << "  Pairwise Gaussian Parameter (x,y,wt):  "
      << x_gauss_ << "  " << y_gauss_ << "  " << wt_gauss_ << std::endl
      << "  Pairwise Bilateral Parameter (x,y,wt):  "
      << x_bilateral_ << "  " << y_bilateral_ << "  " << wt_bilateral_ << std::endl
      << "  Pairwise Color Bilateral Parameter (r,g,b):  "
      << r_bilateral_ << "  " << g_bilateral_ << "  " << b_bilateral_;
}

template <typename Dtype>
void DenseCRFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();

  CHECK_EQ(num, bottom[1]->num());
  CHECK_EQ(height, bottom[1]->height());
  CHECK_EQ(width, bottom[1]->width());
  
  // Syntax : Unaries are the first bottom and image is second bottom
  // Assume that umber of classes in image would be more than channels, just to ensure that order of bottoms was given correct
  CHECK_GE(channels, bottom[1]->channels()) << "Syntax: Pass 'score blob' as first and 'image blob' as second bottom";

  // Output is single-channel image
  top[0]->Reshape(num, 1, height, width);
}

template <typename Dtype>
void DenseCRFLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  const Dtype* bottom_score = bottom[0]->cpu_data();
  const Dtype* bottom_image = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  const int num = bottom[0]->num();
  const int W = bottom[1]->width();
  const int H = bottom[1]->height();
  const int M = bottom[0]->channels();  // No. of Classes
  const int Ch = bottom[1]->channels();  // No. of Image Channels

  unsigned char * im = new unsigned char[ W*H*Ch ];
  float * unary = new float[ W*H*M ];

  // LOG(INFO) << "\n\nDebugging Info for Image Sample, Channels="<<Ch;
  for (int i = 0; i < num; ++i) {

    // Note : 
    // Caffe : Num-Channel-Height-Width in nested for loop outer-to-inner order and BGR for image channels
    // DenseCRF : Num-Height-Width-Channel in nested for loop outer-to-inner order and RGB for image channels
    for( int k=0; k<W*H; k++ ) {
      
      unsigned char * imTemp = im + k*Ch;
      if (Ch==1)
        imTemp[0] = (unsigned char) bottom_image[i * W*H*Ch + 0 * W*H + k];
      else if (Ch==3)   //BGR (Caffe) to RGB (DenseCRF)
      {
        imTemp[0] = (unsigned char) bottom_image[i * W*H*Ch + 2 * W*H + k];
        imTemp[1] = (unsigned char) bottom_image[i * W*H*Ch + 1 * W*H + k];
        imTemp[2] = (unsigned char) bottom_image[i * W*H*Ch + 0 * W*H + k];
      }
      else
        LOG(FATAL) << "Invalid number of Image Channels"<< std::endl;
      
      // if (k<=50 && Ch==3)
      //   LOG(INFO) << "Image Float Values (pixel "<<k<<" , 3-Channelled rgb) : "<< bottom_image[i * W*H*Ch + 2 * W*H + k] <<" "<<bottom_image[i * W*H*Ch + 1 * W*H + k]<<" "<<bottom_image[i * W*H*Ch + 0 * W*H + k]<<std::endl<<"Image Char Converted Values (pixel "<<k<<" , 3-Channelled rgb) : "<<(int)imTemp[0]<<" "<<(int)imTemp[1]<<" "<<(int)imTemp[2]<<std::endl;
      // else if (k<=50 && Ch==1)
      //   LOG(INFO) << "Image Float Values (pixel "<<k<<" , 1-Channelled value) : "<< bottom_image[i * W*H*Ch + 0 * W*H + k]<<std::endl<<"Image Char Converted Values (pixel "<<k<<" , 1-Channelled value) : "<<imTemp[0]<<std::endl;

      float * unaryTemp = unary + k*M;
      for( int j=0; j<M; j++ ) {
        unaryTemp[j] = - (float) bottom_score[i * W*H*M + j * W*H + k];   //Note: Negation b/c densecrf takes it as energy (Lower the better)
        // if (k<=2)
        //   LOG(INFO)<<"Unary Score(pixel "<<k<<" , channel "<<j<<" , rgb) : "<<unaryTemp[j];
      }

    }
    
    DenseInference( im, unary, top_data + i*W*H*1, W, H, M);

    // LOG(INFO) << "\nDebugging Info for Saved Output Score ";
    // for( int k=0; k<W*H*1; k++ ){
      // if (k<=50)
        // LOG(INFO)<<"Predicted Label(pixel "<<k<<" , label) : "<<top_data[i * W*H*1 + 0 * W*H + k];
    // }

  }

  delete[] im;
  delete[] unary;

}

#ifdef CPU_ONLY
STUB_GPU(DenseCRFLayer);
#endif

INSTANTIATE_CLASS(DenseCRFLayer);
REGISTER_LAYER_CLASS(DenseCRF);
}  // namespace caffe
