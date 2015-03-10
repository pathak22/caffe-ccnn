#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/common_layers.hpp"
#include "caffe/layer.hpp"

namespace caffe {

template <typename Dtype>
void RunningAverageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  alpha_ = this->layer_param_.running_average_param().alpha();
  float i_val = this->layer_param_.running_average_param().initial_value();
  CHECK_LE(alpha_, 1) << " alpha cannot be greater than 1.";
  average_.Reshape(1, bottom[0]->channels(),
                   bottom[0]->height(), bottom[0]->width());
  FillerParameter filler_param;
  filler_param.set_type("constant");
  filler_param.set_value(i_val);
  GetFiller<Dtype>(filler_param)->Fill(&average_);
}

template <typename Dtype>
void RunningAverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->channels(), average_.channels()) << " channels match";
  CHECK_EQ(bottom[0]->height(), average_.height()) << " height matches";
  CHECK_EQ(bottom[0]->width(), average_.width()) << " width matches";
  top[0]->ReshapeLike(average_);
}

template <typename Dtype>
void RunningAverageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* average_data = average_.mutable_cpu_data();
  int num = bottom[0]->num();
  int dim = average_.count();
  caffe_scal(dim, alpha_, average_data);
  for (int i = 0; i < num; i++)
    caffe_add(dim, average_data, bottom_data+i*dim, average_data);
  caffe_copy(dim, average_data, top_data);
}
template <typename Dtype>
void RunningAverageLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) return;
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  int num = bottom[0]->num();
  int dim = average_.count();
  for (int i = 0; i < num; i++)
    caffe_copy(dim, top_diff, bottom_diff+i*dim);
}

#ifdef CPU_ONLY
STUB_GPU(RunningAverageLayer);
#else

template <typename Dtype>
void RunningAverageLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  Dtype* average_data = average_.mutable_gpu_data();
  int num = bottom[0]->num();
  int dim = average_.count();
  caffe_gpu_scal(dim, alpha_, average_data);
  for (int i = 0; i < num; i++)
    caffe_gpu_add(dim, average_data, bottom_data+i*dim, average_data);
  caffe_gpu_memcpy(dim*sizeof(Dtype), average_data, top_data);
}
template <typename Dtype>
void RunningAverageLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) return;
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  int num = bottom[0]->num();
  int dim = average_.count();
  for (int i = 0; i < num; i++)
    caffe_gpu_memcpy(dim*sizeof(Dtype), top_diff, bottom_diff+i*dim);
}

#endif

INSTANTIATE_CLASS(RunningAverageLayer);
REGISTER_LAYER_CLASS(RunningAverage);

}  // namespace caffe
