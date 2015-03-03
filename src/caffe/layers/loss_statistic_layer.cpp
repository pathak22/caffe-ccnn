#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
class HardmaxLayer: public Layer<Dtype> {
 public:
  explicit HardmaxLayer(const LayerParameter& param)
     : Layer<Dtype>(param) {}
  void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(),
        bottom[0]->height(), bottom[0]->width());
    mx_.Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  }
  void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* mx_data = mx_.mutable_cpu_data();
    Dtype* imx_data = mx_.mutable_cpu_diff();
    int num = bottom[0]->num();
    int channels = bottom[0]->channels();
    int dim = bottom[0]->count() / bottom[0]->num();
    int spatial_dim = bottom[0]->height() * bottom[0]->width();
    caffe_set(top[0]->count(), Dtype(0), top_data);
    for (int i = 0; i < num; ++i) {
      // initialize scale_data to the first plane
      caffe_copy(spatial_dim, bottom_data + i * dim, mx_data);
      caffe_set(spatial_dim, Dtype(0), imx_data);
      for (int j = 0; j < channels; j++)
        for (int k = 0; k < spatial_dim; k++)
          if (mx_data[k] < bottom_data[i * dim + j * spatial_dim + k]) {
            imx_data[k] = j;
            mx_data[k] = bottom_data[i * dim + j * spatial_dim + k];
          }
    }
  }
  void Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    // Not supported
    if (propagate_down[0])
      LOG(FATAL) << "Cannot backprop through a HardmaxLayer";
  }
  Blob<Dtype> mx_;
};


template <typename Dtype>
void LossStatisticLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  has_ignore_label_ =
    this->layer_param_.loss_statistic_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_statistic_param().ignore_label();
  }
  const bool soft_max = this->layer_param_.loss_statistic_param().max_type() ==
                        LossStatisticParameter_MaxType_SOFT_MAX;
  if (soft_max)
    max_layer_ = shared_ptr<Layer<Dtype> >(
        new SoftmaxLayer<Dtype>(this->layer_param_));
  else
    max_layer_ = shared_ptr<Layer<Dtype> >(
        new HardmaxLayer<Dtype>(this->layer_param_));
  max_bottom_vec_.clear();
  max_bottom_vec_.push_back(bottom[0]);
  max_top_vec_.clear();
  max_top_vec_.push_back(&prob_);
  max_layer_->SetUp(max_bottom_vec_, max_top_vec_);
}

template <typename Dtype>
void LossStatisticLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num()) <<
    "Bottom num's need to match";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width()) <<
    "Bottom width's need to match";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height()) <<
    "Bottom height's need to match";
  int nc = bottom[0]->channels();
  top[0]->Reshape(bottom[0]->num(), 1, nc, nc);
}

template <typename Dtype>
void LossStatisticLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  max_layer_->Forward(max_bottom_vec_, max_top_vec_);

  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int num = prob_.num(), nc = bottom[0]->channels();
  const int dim = prob_.count() / num;
  const int spatial_dim = prob_.height() * prob_.width();
  caffe_set(num * nc * nc, Dtype(0), top_data);

  for (int i = 0; i < num; i++)
    for (int j = 0; j < spatial_dim; j++) {
      const int label_value = static_cast<int>(label[i * spatial_dim + j]);
      if (has_ignore_label_ && label_value == ignore_label_)
        continue;
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, nc);
      for (int c = 0; c < nc; c++)
        top_data[i * nc * nc + label_value * nc + c] +=
          prob_data[i * dim + c * spatial_dim + j];
    }
}

template <typename Dtype>
void LossStatisticLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    const Dtype* label = bottom[1]->cpu_data();
    const Dtype* top_diff = top[0]->mutable_cpu_data();
    Dtype* prob_diff = prob_.mutable_cpu_diff();
    caffe_set(prob_.count(), Dtype(0), prob_diff);
    const int num = prob_.num(), nc = bottom[0]->channels();
    const int dim = prob_.count() / num;
    const int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; i++)
      for (int j = 0; j < spatial_dim; j++) {
        const int label_value = static_cast<int>(label[i * spatial_dim + j]);
        if (!has_ignore_label_ || label_value != ignore_label_) {
          for (int c = 0; c < bottom[0]->channels(); ++c)
            prob_diff[i * dim + c * spatial_dim + j] +=
              top_diff[i * nc * nc + label_value * nc + c];
        }
      }
    max_layer_->Backward(max_bottom_vec_, vector<bool>(1, true),
                         max_top_vec_);
  }
}

#ifdef CPU_ONLY
STUB_GPU(LossStatisticLayer);
#endif

INSTANTIATE_CLASS(LossStatisticLayer);
REGISTER_LAYER_CLASS(LossStatistic);

}  // namespace caffe
