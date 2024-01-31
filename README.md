# 实验五：多模态情感分析

### 运行环境



### 代码文件结构

```
│  ablation.py							消融实验
│  coattentionfusion.py					双流融合模型
│  CoattentionTransformerLayer.py		协同注意力TransormerEncoder
│  concatfusion.py						简单concat模型
│  gatedfusion.py						Gated模型
│  img_train.py							训练resnet50
│  predict.py							查看各模型准确率
│  predict_without_label.py				输出预测结果
│  README.md					
│  text&tag_train.py					训练Bert
│  transformerfusion.py					单流融合模型
│  报告.pdf
│  
├─bert-base-multilingual-cased
│      .gitattributes
│      config.json
│      flax_model.msgpack
│      model.safetensors
│      pytorch_model.bin
│      README.md
│      tf_model.h5
│      tokenizer.json
│      tokenizer_config.json
│      vocab.txt
│      
└─数据
    │  test_without_label.txt
    │  train.txt
    │  
    ├─data
```

### 执行流程

#### 训练

```
python img_train.py
python text&tag_train.py
```

#### 消融实验

```
python ablation.py
```

#### 预测

```
python predict.py --model_type concat/gated/single-stream/double-stream
```

#### 输出结果

```
python predict_without_label.py --model_type concat/gated/single-stream/double-stream
```

### 参考

- T. Zhu, L. Li, J. Yang, S. Zhao, H. Liu and J. Qian, "Multimodal Sentiment Analysis With Image-Text Interaction Network," in IEEE Transactions on Multimedia, vol. 25, pp. 3375-3385, 2023, doi: 10.1109/TMM.2022.3160060.
  keywords: {Sentiment analysis;Visualization;Semantics;Feature extraction;Electronic mail;Social networking (online);Convolutional neural networks;Image-text interaction;multimodal sentiment analysis;region-word alignment},