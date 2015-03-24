#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
__global__ void HardmaxLayer_Forward_dev(const int num, const int channels,
    const int spatial_dim, const Dtype* data, Dtype* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    Dtype maxval = data[(n * channels + 0) * spatial_dim + s];
    int maxind = 0;
    for (int c = 0; c < channels; ++c) {
      Dtype v = data[(n * channels + c) * spatial_dim + s];
      if (v > maxval) {
        maxval = v;
        maxind = c;
      }
    }
    out[(n * channels + maxind) * spatial_dim + s] = 1;
  }
}

template<typename Dtype>
void HardmaxLayer_Forward_gpu(const vector<Blob<Dtype>*>& bottom,
  const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int num = top[0]->num();
  int channels = top[0]->channels();
  int spatial_dim = top[0]->height()*top[0]->width();
  caffe_gpu_set(top[0]->count(), Dtype(0), top_data);
  // NOLINT_NEXT_LINE(whitespace/operators)
  HardmaxLayer_Forward_dev<Dtype><<<CAFFE_GET_BLOCKS(num * spatial_dim),
      CAFFE_CUDA_NUM_THREADS>>>(num, channels, spatial_dim, bottom_data,
      top_data);
}
template void HardmaxLayer_Forward_gpu<float>(const vector<Blob<float>*>&,
  const vector<Blob<float>*>& top);
template void HardmaxLayer_Forward_gpu<double>(const vector<Blob<double>*>&,
  const vector<Blob<double>*>& top);
} // Namespace caffe