#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/common_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class RunningAverageLayerTest : public ::testing::Test {
 protected:
  RunningAverageLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 20, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
    const Dtype * p_bottom = blob_bottom_->cpu_data();
    sum_values_.resize( blob_bottom_->channels(), 0 );
    for( int i=0; i<blob_bottom_->num(); i++ )
      for( int c=0; c<blob_bottom_->channels(); c++ )
        sum_values_[c] += p_bottom[i*blob_bottom_->channels()+c];
  }
  virtual ~RunningAverageLayerTest() { delete blob_bottom_; delete blob_top_; }
  std::vector<Dtype> sum_values_;
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RunningAverageLayerTest, TestDtypes);

TYPED_TEST(RunningAverageLayerTest, TestSetup) {
  LayerParameter layer_param;
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->num(), 1);
}

TYPED_TEST(RunningAverageLayerTest, TestCPU) {
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(top_data[i], this->sum_values_[i]);
  }
}

TYPED_TEST(RunningAverageLayerTest, TestCPUMultiple) {
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for( int i=0; i<10; i++ )
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_NEAR(top_data[i], 10*this->sum_values_[i],1e-4);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestCPUAlpha) {
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(0.5);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for( int i=0; i<50; i++ )
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_NEAR(top_data[i], 2*this->sum_values_[i],1e-5);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestCPUInitialValue) {
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(1.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(top_data[i], this->sum_values_[i]+1);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestCPUBackward) {
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(1.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_set(this->blob_top_->count(), TypeParam(1),
    this->blob_top_->mutable_cpu_diff() );
  layer.Backward(this->blob_top_vec_, std::vector<bool>(1,true),
    this->blob_bottom_vec_);
  // Now, check values
  const TypeParam* bottom_diff = this->blob_bottom_->cpu_diff();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(bottom_diff[i], 1);
  }
}


TYPED_TEST(RunningAverageLayerTest, TestGPU) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(top_data[i], this->sum_values_[i]);
  }
}

TYPED_TEST(RunningAverageLayerTest, TestGPUMultiple) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for( int i=0; i<10; i++ )
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_NEAR(top_data[i], 10*this->sum_values_[i],1e-4);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestGPUAlpha) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(0.5);
  running_average_param->set_initial_value(0.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  for( int i=0; i<50; i++ )
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_NEAR(top_data[i], 2*this->sum_values_[i],1e-5);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestGPUInitialValue) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(1.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  // Now, check values
  const TypeParam* top_data = this->blob_top_->cpu_data();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(top_data[i], this->sum_values_[i]+1);
  }
}
TYPED_TEST(RunningAverageLayerTest, TestGPUBackward) {
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  RunningAverageParameter* running_average_param = 
    layer_param.mutable_running_average_param();
  running_average_param->set_alpha(1.0);
  running_average_param->set_initial_value(1.0);
  RunningAverageLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  caffe_set(this->blob_top_->count(), TypeParam(1),
    this->blob_top_->mutable_cpu_diff() );
  layer.Backward(this->blob_top_vec_, std::vector<bool>(1,true),
    this->blob_bottom_vec_);
  // Now, check values
  const TypeParam* bottom_diff = this->blob_bottom_->cpu_diff();
  int channels = this->blob_top_->channels();
  for (int i = 0; i < channels; ++i) {
    EXPECT_FLOAT_EQ(bottom_diff[i], 1);
  }
}

}  // namespace caffe
