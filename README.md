# Emotional-Chatbot

최근 AI의 발전으로 많은 AI를 활용한 많은 서비스가 출품되고 있다. 기술의 발전으로 인해 AI는 우리의 삶에 더 많은 영향을 끼치고 있으며, 특히 AI 와의 대화(소통)이 증가하고 있다. 인간의 대화에서는 대화의 감정과 맥락을 이해하고 상황에 맞는 적절한 대답을 선택하는 것이 중요하다. 하지만, AI는 대부분 문맥을 기반으로 작동하며, 사전에 학습된 데이터와 패턴을 활용하여 응답을 생성한다. 이로 인해 현재 Chatbot과 같은 AI는 감정을 완벽하게 해석하거나 대응하는 데 어려움을 겪을 수 있다. 따라서, 감정과 대화를 기반으로 KoGPT를 학습하여 감정을 이해하는 ChatBot을 만드는 것이 이번 프로젝트의 목표이다.  

## Data Preprocessing
- AI 허브에서 제공하는 "감성대화말뭉치" 데이터 세트를 사용
  - 60가지의 세부 감정에 대한 말뭉치 대화 데이터 세트
  - 대화는 총 60가지 감정으로 라벨링 되어 있음

## Model 1
 - KoBERT(Pretrained on base-v1)

## Model 2
 - KoGPT2

## Model 3
- 1) [Question]을 통해 [감정_소분류]를 추론한다.
- 2) 예측한 [감정_소분류] 와 [Question] 을 통해 [Answer]를 추론한다
  - 이 때, 감정 토큰에 대한 Forced Attention 기법 도입




---
```
# # 라이브러리 설치
# !pip install gluonnlp pandas tqdm
# !pip install mxnet
# !pip install sentencepiece
# !pip install transformers
# !pip install torch
# !pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

# 사용한 아나콘다 가상 환경
# NLP (Python 3.9.16)

# pip install mxnet-mkl==1.6.0 numpy==1.23.1
# pip install openpyxl
```