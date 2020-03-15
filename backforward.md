# 反向传播层
1. Loss Layer 损失层是caffe CNN的终点，损失函数（loss function， 也可以称作）可以分为两大类：分类问题的损失函数与回归问题的损失函数，常用的损失函数包括：
    - softMax cross entropy loss（softmax 交叉熵损失函数）：它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类！
    - cross entropy loss （交叉熵损失函数）
    - Mean Square Loss （MSL，均方误差）

2. softMax 源码详解（参考http://onlynice.me/2018/03/03/%E5%9F%BA%E4%BA%8ELeNet%E7%BD%91%E7%BB%9C%E7%9A%84Caffe%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90-SoftmaxLoss%E5%B1%82/）

-   caffe将softmax loss分成了softmaxLayer 和softmaxWithLossLayer两层
- 在Caffe中有一个概念即为softmax_axis,即沿着哪一个轴进行softmax计算。一般Caffe满足的是N×C×H×W的计算顺序，一般达到最后的层时，例如mnist，H,W都为1，即最终的数据维度为:N×C×1×1，默认的softmax axis是1,即C对应的轴

- 前项传播
```
template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  Dtype* scale_data = scale_.mutable_cpu_data();
  // 选择softmax中的axis，从softmax的公式中我们可以get到需要在某一个维度进行求和
  int channels = bottom[0]->shape(softmax_axis_);

   // dim 表示每张图片的维度（所占的长度）
  int dim = bottom[0]->count() / outer_num_;
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  // We need to subtract the max to avoid numerical issues, compute the exp,
  // and then normalize.
  //最外边的for循环的outer_num_对应的是样本的个数。此外，inner_num_对应的是H×W。bottom的数据是存储在Blob的一个连续空间部分，所以，对于不同的样本使用i∗dim来移动指针
  for (int i = 0; i < outer_num_; ++i) {
    // initialize scale_data to the first plane
    caffe_copy(inner_num_, bottom_data + i * dim, scale_data);
    for (int j = 0; j < channels; j++) {
    //for的关键作用就是求取最大值，用来后边减去最大值防止数值溢出，每个位置都要计算10个通道的softmax值
      for (int k = 0; k < inner_num_; k++) {
        scale_data[k] = std::max(scale_data[k],
            bottom_data[i * dim + j * inner_num_ + k]);
      }
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_,
        1, -1., sum_multiplier_.cpu_data(), scale_data, 1., top_data);
    // exponentiation
    caffe_exp<Dtype>(dim, top_data, top_data);
    // sum after exp
    caffe_cpu_gemv<Dtype>(CblasTrans, channels, inner_num_, 1.,
        top_data, sum_multiplier_.cpu_data(), 0., scale_data);
    // division
    for (int j = 0; j < channels; j++) {
      caffe_div(inner_num_, top_data, scale_data, top_data);
      top_data += inner_num_;
    }
  }
}
```

- 反向传播

```
对softmax进行求导，根据求导后公式，进行一系列计算，具体推到过程可以参考（http://onlynice.me/2018/03/03/%E5%9F%BA%E4%BA%8ELeNet%E7%BD%91%E7%BB%9C%E7%9A%84Caffe%E4%BB%A3%E7%A0%81%E5%88%86%E6%9E%90-SoftmaxLoss%E5%B1%82/）
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  const Dtype* top_data = top[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* scale_data = scale_.mutable_cpu_data();
  int channels = top[0]->shape(softmax_axis_);
  int dim = top[0]->count() / outer_num_;
  caffe_copy(top[0]->count(), top_diff, bottom_diff);
  for (int i = 0; i < outer_num_; ++i) {
    // compute dot(top_diff, top_data) and subtract them from the bottom diff
    for (int k = 0; k < inner_num_; ++k) {
      scale_data[k] = caffe_cpu_strided_dot<Dtype>(channels,
          bottom_diff + i * dim + k, inner_num_,
          top_data + i * dim + k, inner_num_);
    }
    // subtraction
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, inner_num_, 1,
        -1., sum_multiplier_.cpu_data(), scale_data, 1., bottom_diff + i * dim);
  }
  // elementwise multiplication
  caffe_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}
```

- SoftmaxWithLossLayer， lossLayer是从softmax的top层取结果，进行loss的函数计算
   -forward
   ```
   template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    // The forward pass computes the softmax prob values.
    softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
    const Dtype* prob_data = prob_.cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    Dtype loss = 0;
    for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; j++) {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
            if (has_ignore_label_ && label_value == ignore_label_) {
                continue;
            }
            DCHECK_GE(label_value, 0);
            DCHECK_LT(label_value, prob_.shape(softmax_axis_));
            loss -= log(std::max(prob_data[i * dim + label_value * inner_num_ + j],Dtype(FLT_MIN)));
            ++count;
        }
    }
    top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);
    if (top.size() == 2) {
        top[1]->ShareData(prob_);
    }
    }
   ```
   - backward
   对loss function进行求导
   ```
   template <typename Dtype>
    void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                << " Layer cannot backpropagate to label inputs.";
    }
    if (propagate_down[0]) {
        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* prob_data = prob_.cpu_data();
        caffe_copy(prob_.count(), prob_data, bottom_diff);
        const Dtype* label = bottom[1]->cpu_data();
        int dim = prob_.count() / outer_num_;
        int count = 0;
        for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; ++j) {
            const int label_value = static_cast<int>(label[i * inner_num_ + j]);
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
        Dtype loss_weight = top[0]->cpu_diff()[0] /
                            get_normalizer(normalization_, count);
        caffe_scal(prob_.count(), loss_weight, bottom_diff);
    }
    }

   ```
