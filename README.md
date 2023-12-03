# Emotional-Chatbot

최근 몇 년 동안, 자연어처리와 인공지능 기술의 발전은 기계와 인간간의 소통을 혁신적으로 증가시켰다. 이러한 발전의 중심에는 Chat-GPT와 같은 대화형 모델이 있으며, 인간과 기계 간의 대화를 보다 효과적으로 가능하게 한다 인간의 대화에서는 대화의 감정과 맥락을 이해하고 상황에 맞는 적절한 대답을 선택하는 것이 중요하다. 하지만, AI는 대부분 문맥을 기반으로 작동하며, 사전에 학습된 데이터와 패턴을 활용하여 응답을 생성한다. 이로 인해 현재 Chatbot과 같은 AI는 감정을 완벽하게 해석하거나 대응하는 데 어려움을 겪을 수 있다. 따라서, 감정과 대화를 기반으로 KoGPT를 학습하여 감정을 이해하는 ChatBot을 만드는 것이 이번 프로젝트의 목표이다.  

## Data Preprocessing
- [Data Preprocessing README.md](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/Data_preprocessing/README.md)

## Question 2 Emotion
 - [Question-Emotion_Training README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/Question-Emotion_Training/README.md)

 - KoBERT(Pretrained on base-v1)
   - [Question] 을 통해 [감정] 를 학습한다.
   - [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/1V4v0ppYLoDvwemRnVpd-0QCYnCnqDSsl)

## (Inferenced)Emotion + Question 2 Answer
 - [EmotionQ-Answer_Training README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/EmotionQ-Answer_Training/README.md)

 - KoGPT2
   - [감정]과 [Question] 을 통해 [Answer]를 학습한다.
   - [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/13MgcxhXt_BPmEg9-LK1y8Af2gPoBrRI2)

## ChatBot
- [FinalModel README.md](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/FinalModel/README.md)

- 1) [Question]을 통해 [감정]을 추론한다.

- 2) 예측한 [감정]과 [Question] 을 통해 [Answer]를 추론한다.
    - [Pre-Trained Weight](https://drive.google.com/drive/u/0/folders/1XFDGbr1ATrh1g_LSEyxZy5arHYFVc50M)

---
### SETTING

#### Computing environment
- OS: Ubuntu 20.04.6 LTS
- GPU : RTX 4090

#### Version
- Python : 3.9.16
- CUDA : 11.8
- CuDNN : 8.6.0
- torch : 2.1.0
- torchvision : 0.16.0
- transformers : 4.35.2



#### EmotionQ-Answer_Training
```bash
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install pytorch-lightning
```

#### FinalModel
```bash
```