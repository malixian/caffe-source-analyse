# Caffe 源码分析

## Caffe的核心数据结构
- Blob 
- Layer
- Net
- Solver

## caffe的一些核心依赖包

- CUDA
- BLAS：高性能的现代运算库
- OpenCV
- Google：protobuffer、gfalgs、glog
- IO：Hdf5、leveldb、lmdb

## 核心数据结构解析
- Blob
Blob 抽象出了神经网络图中流动的数据，可以类似于tensorflow中的tensor数据结构
Blob类中的核心函数：
1. 通过构造函数 `explicit Blob(const vector<int>& shape); ` 确定数据的维度，构造函数中通过explicit避免隐使得转换。维度的特征可以从`explicit Blob(const int num, const int channels, const int height, const int width);`中明显的看出来，包含num（即为batch size）、channels、height、width
2. Reshape()函数，根据给定的参数改变输入blob的维度，仅改变维度但是内容不变，caffe中的四维数据通过一维的数组来表示，并使用offset内联函数来确定每个数据的位置`((n * channels() + c) * height() + h) * width() + w`
3. 一些核心的成员函数
```
  const Dtype* cpu_data() const; //cpu使用的数据
  void set_cpu_data(Dtype* data);//用数据块的值来blob里面的data。
  const Dtype* gpu_data() const;//返回不可更改的指针，下同
  const Dtype* cpu_diff() const;
  const Dtype* gpu_diff() const;
  Dtype* mutable_cpu_data();//返回可更改的指针，下同
  Dtype* mutable_gpu_data();
  Dtype* mutable_cpu_diff();
  Dtype* mutable_gpu_diff();
```
带mutable_开头的意味着可以对返回的指针内容进行更改，而不带mutable_开头的返回const 指针，不能对其指针的内容进行修改

4. 成员变量
```
  shared_ptr<SyncedMemory> data_;//原始数据
  shared_ptr<SyncedMemory> diff_;//梯度数据
  shared_ptr<SyncedMemory> shape_data_;
  vector<int> shape_;//维度信息
  int count_;//为blob的size，即总容量
  int capacity_;//类比容器就是超过容量需要重新分配内存
```
5. c++中几种类型转换
- static_cast	编译期间，用于良性转换，一般不会导致意外发生，风险很低。向上转换无信息丢失，向下转换可能有信息丢失
- const_cast	用于 const 与非 const、volatile 与非 volatile 之间的转换。
- reinterpret_cast	高度危险的转换，这种转换仅仅是对二进制位的重新解释，不会借助已有的转换规则对数据进行调整，但是可以实现最灵活的 C++ 类型转换。
- dynamic_cast	借助 RTTI，用于类型安全的向下转型（Downcasting）。