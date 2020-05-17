# Transformer Network

한국법제연구원 대한민국 영문법령 데이터셋과 Pytorch로 구현한 Transformer 기반 번역기 입니다.

* Korean Blog: [Annotated Transformer KR](https://www.notion.so/simonjisu/Attention-Is-All-You-Need-5944fbf370ab46b091eeb64453ac3af5)

# Getting Started

### 모델과 데이터에 대한 설명
본 모델은 Mecab(한글 형태소 분석기)과 Spacy(영문 형태소 분석기)를 이용한 트랜스포머 번역 모델입니다.
본 모델에 사용된 데이터는 ./data/korean-parallel-corpora/koen 경로 내 train / valid / test 파일들로 구성되어 있습니다.

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
