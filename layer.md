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
    -Forward：前向传播函数
    -Backward：反向传播函数
    -LayerSetUp：初始化layer