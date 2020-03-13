## layer 数据结构解析
- layer层的实现类型非常多，我们通过分析最原始的layer层，来查看layer的核心功能
1. 成员变量
```
/** The protobuf that stores the layer parameters */
  LayerParameter layer_param_;
  /** The phase: TRAIN or TEST */
  /*
  
  */
  Phase phase_;
  /** The vector that stores the learnable parameters as a set of blobs. */
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  /** Vector indicating whether to compute the diff of each param blob. */
  vector<bool> param_propagate_down_;

  /** The vector that indicates whether each top blob has a non-zero weight in
   *  the objective function. */
  vector<Dtype> loss_;

```
2. 主要的函数
    -Forward：前向传播函数inline函数，包括核心的虚函数Forward_cpu、Forward_gpu
    -Backward：反向传播函数inline函数，包括核心的虚函数Backward_cpu、Backward_gpu
    -LayerSetUp：初始化layer

3. layer 工厂函数
- layer同过多态实现了非常多种layer，由此产生的问题是在用户使用到子类的时候不得不使用到new XX的代码，因此客户端必须知道子类的名称，而且当类明修改时需要修改全部的new XX的代码。通过工厂模式可以改变此问题
- 工厂模式的分类：简单工厂（simple factory）、工厂方法(factory method)、抽象工厂(abstrct factory),此三种工厂模式的复杂度依次递增，根据业务需求选择不同的工厂模式。

4. 以卷积层为例进行分析
- 父子关系：layer->base_conv_layer->conv_layer
- base_conv_layer成员变量
```
 /// @brief The spatial dimensions of a filter kernel.
  // kernel的形状 = [kernel_h, kernel_w]
  Blob<int> kernel_shape_;

  /// @brief The spatial dimensions of the stride.
  // 步长形状 = [stride_h, stride_w]
  Blob<int> stride_;

  /// @brief The spatial dimensions of the padding.
  // pad的形状 = [pad_h, pad_w]
  Blob<int> pad_;

  /// @brief The spatial dimensions of the dilation.
  Blob<int> dilation_;

  /// @brief The spatial dimensions of the convolution input.
  // 卷积的输入形状 = [输入图像通道数, 输入图像h,    输入图像w]
  Blob<int> conv_input_shape_;

  /// @brief The spatial dimensions of the col_buffer.
  // col_buffer的形状 = [kernel_dim_, conv_out_spatial_dim_ ]
  vector<int> col_buffer_shape_;

  /// @brief The spatial dimensions of the output.
  // 输出的形状
  vector<int> output_shape_;

  // 输入的形状
  const vector<int>* bottom_shape_;

  //空间轴个数
  int num_spatial_axes_;
  // 输入度维度 = 输入图像通道数*输入图像的h*输入图像w
  int bottom_dim_;
  // 输出度维度 = 输出图像通道数*输出图像的h*输出图像w
  int top_dim_;

  // 输入图像的第几个轴是通道
  int channel_axis_;
  // batch size
  int num_;
  // 输入图像的通道数
  int channels_;
  // 卷积组的大小
  int group_;
  // 输出空间维度 = 卷积之后的图像长*卷积之后图像的宽
  int out_spatial_dim_;
  // 使用卷积组用到的
  int weight_offset_;
  // 卷积后的图像的通道数
  int num_output_;
  // 是否启用偏置
  bool bias_term_;
  // 是不是1x1卷积
  bool is_1x1_;
  // 强制使用n维通用卷积
  bool force_nd_im2col_;
  // conv_in_channels_ * conv_out_spatial_dim_
  int num_kernels_im2col_;
  // num_kernels_col2im_ = reverse_dimensions() ? top_dim_ : bottom_dim_
  int num_kernels_col2im_;
 
  // 卷积的输出通道数 ,在参数配置文件中设置
  int conv_out_channels_;
 
  // 卷积的输入通道数 （即输入图像的通道数）
  int conv_in_channels_;
 
  // 卷积的输出的空间维度 = 卷积后图像h*卷积后图像w
  int conv_out_spatial_dim_;
 
  // 卷积核的维度 = 输入图像的维度*卷积核的h*卷积核的w
  int kernel_dim_;
 
  // 在使用gropu参数的时候使用的offset
  int col_offset_;
  int output_offset_;
  
  // im2col的时候使用的存储空间
  Blob<Dtype> col_buffer_;
 
  // 将偏置扩展成矩阵的东东
  Blob<Dtype> bias_multiplier_;
```
- 成员函数，ConvolutionLayer是继承于BaseConvolutionLayer的，而BaseConvolutionLayer才是真正实现卷积及其反传的，而在BaseConvolutionLayer中的卷积的实现中有一个重要的函数就是im2col以及col2im，im2colnd以及col2imnd。前面的两个函数是二维卷积的正向和逆向过程，而后面的两个函数是n维卷积的正向和逆向过程。有关这四个函数的详细过程可以参考 (https://blog.csdn.net/xizero00/article/details/51049858)
    - conv_layer.cpp中的forward代码，我们从外至内逐步剖析代码
```
    template <typename Dtype>
    // 输入bottom层，输出top层
    void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
          const vector<Blob<Dtype>*>& top) {
      const Dtype* weight = this->blobs_[0]->cpu_data();
      for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->cpu_data();
        Dtype* top_data = top[i]->mutable_cpu_data();
        // num_ = batchsize
        for (int n = 0; n < this->num_; ++n) {
          // 基类的forward_cpu_gemm函数
          // 计算的是top_data[n * this->top_dim_] =
          // weights X bottom_data[n * this->bottom_dim_]
          // 输入的是一幅图像的数据，对应的是这幅图像卷积之后的位置
          this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
              top_data + n * this->top_dim_);
          if (this->bias_term_) {
            const Dtype* bias = this->blobs_[1]->cpu_data();
            this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
          }
        }
      }
    }
```
   - BaseConvolutionLayer中的forwad_cpu_gemm函数
```
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_gemm(const Dtype* input,
    const Dtype* weights, Dtype* output, bool skip_im2col) {
  const Dtype* col_buff = input;
  if (!is_1x1_) {
    if (!skip_im2col) {
      // 如果没有1x1卷积，也没有skip_im2col
      // 则使用conv_im2col_cpu对使用卷积核滑动过程中的每一个kernel大小的图像块
      // 变成一个列向量，形成一个height=kernel_dim_的
      // width = 卷积后图像heght*卷积后图像width
      conv_im2col_cpu(input, col_buffer_.mutable_cpu_data());
    }
    col_buff = col_buffer_.cpu_data();
  }
 
  // 使用caffe的cpu_gemm来进行计算
  for (int g = 0; g < group_; ++g) {
      // 分组分别进行计算
      // conv_out_channels_ / group_是每个卷积组的输出的channel
      // kernel_dim_ = input channels per-group x kernel height x kernel width
      // 计算的是output[output_offset_ * g] =
      // weights[weight_offset_ * g] X col_buff[col_offset_ * g]
      // weights的形状是 [conv_out_channel x kernel_dim_]
      // col_buff的形状是[kernel_dim_ x (卷积后图像高度乘以卷积后图像宽度)]
      // 所以output的形状自然就是conv_out_channel X (卷积后图像高度乘以卷积后图像宽度)
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, conv_out_channels_ /
        group_, conv_out_spatial_dim_, kernel_dim_,
        (Dtype)1., weights + weight_offset_ * g, col_buff + col_offset_ * g,
        (Dtype)0., output + output_offset_ * g);
  }
}
 
template <typename Dtype>
void BaseConvolutionLayer<Dtype>::forward_cpu_bias(Dtype* output,
    const Dtype* bias) {
  // output = bias * bias_multiplier_
  // num_output 与 conv_out_channel是一样的
  // num_output_ X out_spatial_dim_ = num_output_ X 1    1 X out_spatial_dim_
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_output_,
      out_spatial_dim_, 1, (Dtype)1., bias, bias_multiplier_.cpu_data(),
      (Dtype)1., output);
}
```
   - conv_im2col_cpu将卷积核在图像上的滑动转换为了矩阵

```
inline void conv_im2col_cpu(const Dtype* data, Dtype* col_buff) {
    if (!force_nd_im2col_ && num_spatial_axes_ == 2) {
      im2col_cpu(data, conv_in_channels_,
          conv_input_shape_.cpu_data()[1], conv_input_shape_.cpu_data()[2],
          kernel_shape_.cpu_data()[0], kernel_shape_.cpu_data()[1],
          pad_.cpu_data()[0], pad_.cpu_data()[1],
          stride_.cpu_data()[0], stride_.cpu_data()[1], col_buff);
    } else {
      im2col_nd_cpu(data, num_spatial_axes_, conv_input_shape_.cpu_data(),
          col_buffer_shape_.data(), kernel_shape_.cpu_data(),
          pad_.cpu_data(), stride_.cpu_data(), col_buff);
    }
  }

```
- 卷积计算的核心函数 im2col，将image图片窗口内的图像（flat）转换为一列 （在util/im2col.cpp中可以看到），通过im2col函数可以将卷积操作变成两个矩阵相乘。此优化是贾扬清对卷积操作的一个优化，正常的卷积操作如下所示
```
  //如果不用这样的操作，贾扬清有一个吐槽，对于输入大小为W*H，维度为D的blob，卷积核为M*K*K，那么如果利用for循环，会是这样的一个操作，6层for循环，计算效率是极其低下的。
  for w in 1..W
 for h in 1..H
   for x in 1..K
     for y in 1..K
       for m in 1..M
         for d in 1..D
           output(w, h, m) += input(w+x, h+y, d) * filter(m, x, y, d)
         end
       end
     end
   end
 end
end

```
- 通过im2col优化后
```
// 输入参数为：im2col_cpu(一幅图像，输入图像的channel, 输入图像的height, 输入图像的width, kernel的height, kernel的width, pad的height, pad的width, stride的height， stride的width)
//dilation:膨胀系数，卷积核膨胀是将卷积核扩张到膨胀尺度约束的尺度中，并将原卷积核没有占用的区域填充零
// 在有膨胀稀疏下，kernel_h = dilation_h * (kernel_h - 1) + 1
void im2col_cpu(const Dtype* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    const int dilation_h, const int dilation_w,
    Dtype* data_col) {
  // 卷积后输出的高度
  const int output_h = (height + 2 * pad_h -
    (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  // 卷积后输出的宽度
  const int output_w = (width + 2 * pad_w -
    (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  // 
  const int channel_size = height * width;
  for (int channel = channels; channel--; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = output_h; output_rows; output_rows--) {
          // 溢出边界补0
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = output_w; output_cols; output_cols--) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = output_w; output_col; output_col--) {
              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                *(data_col++) = data_im[input_row * width + input_col];
              } else {
                 // 溢出边界补0
                *(data_col++) = 0;
              }
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}
```
