# Transformer Network

한국법제연구원 대한민국 영문법령 데이터셋과 Pytorch로 구현한 Transformer 기반 번역기 입니다.

* Korean Blog: [Annotated Transformer KR](https://www.notion.so/simonjisu/Attention-Is-All-You-Need-5944fbf370ab46b091eeb64453ac3af5)

# Getting Started

### 모델과 데이터에 대한 설명
본 모델은 Embedding Weight Sharing(EWS) 모델을 구현한 트랜스포머 번역 모델입니다. Embedding Weight Sharing이란, Source Data(번역할 한글 데이터)와 Target Data(번역될 영문 데이터)가 같은 Embedding Matrix를 공유함으로써 같은 Embedding Space를 사용하여 Language Model을 구현하는 방법입니다.

본 모델에 사용된 데이터는 ./data/korean-parallel-corpora/koen 경로 내 train / valid / test 파일들로 구성되어 있으며, EWS 모델을 구현하기 위해 법령 용어 사전에 mapping되는 한글 법령용어를 영문 법령요어로 대체한 Source Data(번역할 한글 데이터)를 사용하였다는 점이 특징입니다.

특히, 한글 법령용어를 영문 법령용어로 대체한 Corpus(말뭉치)를 Word2Vec의 Skip-gram 방식으로 Pretrain(사전 학습)하여 Embedding Matrix를 초기화 하였으며, 학습 과정에서 Fine Tuning 되도록 구현하였습니다.

### 학습 및 번역 방법
run-koen.sh 파일을 실행하여 학습을 시작하며, 학습을 마치고 난 후에는 translate.py 파일을 실행하여 번역 결과를 확인할 수 있습니다.

### Requirements

```
python >= 3.6
pytorch >= 1.0.0
torchtext
numpy
```

# References

* origin : https://github.com/simonjisu/annotated-transformer-kr
* paper : https://arxiv.org/abs/1706.03762
* reference blog: http://nlp.seas.harvard.edu/2018/04/03/attention.html
* reference code: https://github.com/jadore801120/attention-is-all-you-need-pytorch
