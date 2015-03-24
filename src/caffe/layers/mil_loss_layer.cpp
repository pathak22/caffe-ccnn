#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

bool debug = false;

template <typename Dtype>
void MILLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]); // &adapted_bottom_ : If backprop on updated score but doesn't make sense
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  normalize_ = this->layer_param_.loss_param().normalize();
}

template <typename Dtype>
void MILLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
  adapted_bottom_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MILLossLayer<Dtype>::InferLatentLabel(
    const vector<Blob<Dtype>*>& bottom) {

  latent_label_.clear();
  adapted_bottom_.CopyFrom(*bottom[0]);
  Dtype* adapted_data = adapted_bottom_.mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = outer_num_;
  int dim = adapted_bottom_.count() / outer_num_;
  int spatial_dim = adapted_bottom_.count(softmax_axis_ + 1);
  int channels = adapted_bottom_.shape(softmax_axis_);

  vector<int> im_label;
  Dtype cb,cf,back_perc=0, fore_perc=0, other_perc=0;

  if (debug) { LOG(INFO) <<"\nNew Image :\n"; }
  for (int i = 0; i < num; ++i) {

    // Initialize variables
    int weak_labels[channels];
    for (int k=0;k<channels;k++) {
      weak_labels[k]=0;
    }

    // Find image level labels from segmentation mask 
    // Spatial Dimension of original image is different from heat map
    for (int j = 0; j < bottom[1]->count(softmax_axis_ + 1); j++) {
      int k = static_cast<int>(label[i * bottom[1]->count(softmax_axis_ + 1) + j]);
      if (k == -1 || k == 255) {
        continue;
      }
      weak_labels[k] = 1;
    }

    cb = 1;
    cf = 1;
    int iterNum = 0;
    while(!(back_perc > 40 && fore_perc >30 && other_perc==0) && iterNum<20) {
      iterNum++;
      if (debug) { LOG(INFO) <<"\nLatent Iteration="<<iterNum; }

      im_label.clear();
      back_perc = 0;
      fore_perc = 0;
      other_perc = 0;

      // Computing Latent Labels
      for (int j = 0; j < spatial_dim; j++) {
        int k_max = -1000000;
        Dtype score_max = -1000000;
        if (debug && j==22) { LOG(INFO) <<"Scores for spatial location="<<j; }
        for (int k = 0; k < channels; k++) {
          // Bias the score
          if (k==0)
            adapted_data[i * dim + k * spatial_dim + j] += cb;
          else if (weak_labels[k]==1)
            adapted_data[i * dim + k * spatial_dim + j] += cf;
          Dtype temp = std::max(adapted_data[i * dim + k * spatial_dim + j],Dtype(FLT_MIN));
          if (score_max<temp) {     //  && weak_labels[k]==1 : To only select among image-labels
            score_max = temp;
            k_max = k;
          }
          if (debug && j==22) { LOG(INFO) <<"Score for Class="<<k<<" is "<<temp; }
        }
        if (debug && j==22) { LOG(INFO) <<"Max-Class="<<k_max<<" and max-score is "<<score_max; }
        im_label.push_back(k_max);
        if (k_max==0)
          back_perc++;
        else if (weak_labels[k_max]==1)
          fore_perc++;
        else
          other_perc++;
      }

      // Compute percentage of background and foreground
      back_perc *= 100.0/spatial_dim;
      fore_perc *= 100.0/spatial_dim;
      other_perc *= 100.0/spatial_dim;

      // Update cb and cf
      if (other_perc > 0)
      {
        if (debug) { LOG(INFO) <<"Other_Perc Violated !"; }
        cb = 1; cf = 1;
      } else if (back_perc < 40) {
        if (debug) { LOG(INFO) <<"Back_Perc Violated !"; }
        cb = 1; cf = 0;
      } else if (fore_perc < 20) {
        if (debug) { LOG(INFO) <<"Fore_Perc Violated !"; }
        cb = 0; cf = 1;
      }

      if (debug) { LOG(INFO) <<"Percentage Back="<<back_perc<<" Fore="<<fore_perc<<" Other="<<other_perc; }

    }

    for (int j=0; j<im_label.size(); j++)
      latent_label_.push_back(im_label[j]);

  }
}


template <typename Dtype>
void MILLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Infer latent pixel wise labels 
  InferLatentLabel(bottom);
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(latent_label_[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }
  if (normalize_) {
    top[0]->mutable_cpu_data()[0] = loss / count;
  } else {
    top[0]->mutable_cpu_data()[0] = loss / outer_num_;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void MILLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(latent_label_[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    if (normalize_) {
      caffe_scal(prob_.count(), loss_weight / count, bottom_diff);
    } else {
      caffe_scal(prob_.count(), loss_weight / outer_num_, bottom_diff);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(MILLossLayer);
#endif

INSTANTIATE_CLASS(MILLossLayer);
REGISTER_LAYER_CLASS(MILLoss);

}  // namespace caffe
