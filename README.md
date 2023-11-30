# Emotional-Chatbot

최근 몇 년 동안, 자연어처리와 인공지능 기술의 발전은 기계와 인간간의 소통을 혁신적으로 증가시켰다. 이러한 발전의 중심에는 Chat-GPT와 같은 대화형 모델이 있으며, 인간과 기계 간의 대화를 보다 효과적으로 가능하게 한다 인간의 대화에서는 대화의 감정과 맥락을 이해하고 상황에 맞는 적절한 대답을 선택하는 것이 중요하다. 하지만, AI는 대부분 문맥을 기반으로 작동하며, 사전에 학습된 데이터와 패턴을 활용하여 응답을 생성한다. 이로 인해 현재 Chatbot과 같은 AI는 감정을 완벽하게 해석하거나 대응하는 데 어려움을 겪을 수 있다. 따라서, 감정과 대화를 기반으로 KoGPT를 학습하여 감정을 이해하는 ChatBot을 만드는 것이 이번 프로젝트의 목표이다.  

## Data Preprocessing
- AI 허브에서 제공하는 "감성대화말뭉치" 데이터 세트를 사용
  - 60가지의 세부 감정에 대한 말뭉치 대화 데이터 세트
  - 대화는 총 60가지 감정으로 라벨링 되어 있음

    ![Untitled (9)](https://github.com/hankyuwon/Emotional-Chatbot/assets/98513704/e2c3ef9f-1b71-4aba-a8dd-abad14253b50)

## Model 1
 - KoBERT(Pretrained on base-v1)
   - [Question] 을 통해 [감정_소분류] 를 학습한다.
   - DownLoad Weight and  [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/1V4v0ppYLoDvwemRnVpd-0QCYnCnqDSsl)

## Model 2
 - KoGPT2
   - [감정_소분류]와 [Question] 을 통해 [Answer]를 학습한다.
     - 이 때, 감정 토큰에 대한 Forced Attention 기법 도입
   - [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/13MgcxhXt_BPmEg9-LK1y8Af2gPoBrRI2)

## Model 3
- 1) [Question]을 통해 [감정_소분류]를 추론한다.
- 2) 예측한 [감정_소분류] 와 [Question] 을 통해 [Answer]를 추론한다
<br/>
<br/>
    - [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/1XFDGbr1ATrh1g_LSEyxZy5arHYFVc50M)





---
### SETTING

#### Computing environment
- OS: Ubuntu 18.04.6
- GPU : RTX 4090

#### Version
- Python : 3.9.16
- CUDA : 11.8
- CuDNN : 8.6.0
- torch : 2.1.0
- torchvision : 0.16.0
- transformers : 4.35.2

#### Question-Emotion_Training
```bash
!pip install gluonnlp pandas tqdm
!pip install mxnet
!pip install sentencepiece
!pip install transformers
!pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

#### EmotionQ-Answer_Training
```bash
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install pytorch-lightning
```

#### FinalModel
```bash
```