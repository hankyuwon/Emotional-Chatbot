# Emotional-Chatbot

최근 몇 년 동안, 자연어처리와 인공지능 기술의 발전은 기계와 인간간의 소통을 혁신적으로 증가시켰다. 이러한 발전의 중심에는 Chat-GPT와 같은 대화형 모델이 있으며, 인간과 기계 간의 대화를 보다 효과적으로 가능하게 한다 인간의 대화에서는 대화의 감정과 맥락을 이해하고 상황에 맞는 적절한 대답을 선택하는 것이 중요하다. 하지만, AI는 대부분 문맥을 기반으로 작동하며, 사전에 학습된 데이터와 패턴을 활용하여 응답을 생성한다. 이로 인해 현재 Chatbot과 같은 AI는 감정을 완벽하게 해석하거나 대응하는 데 어려움을 겪을 수 있다. 따라서, 감정과 대화를 기반으로 KoGPT를 학습하여 감정을 이해하는 ChatBot을 만드는 것이 이번 프로젝트의 목표이다.  

## 1. Data Preprocessing
- [Data Preprocessing](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/Data_preprocessing)

## 2. Question 2 Emotion
 - [Question-Emotion_Training](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/Question-Emotion_Training)

 - KoBERT **(Pretrained on skt/kobert-base-v1)**

## 3. (Inferenced)Emotion + Question 2 Answer
 - [EmotionQ-Answer_Training](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/EmotionQ-Answer_Training)

 - KoGPT2 **(Pretrained on skt/kogpt2-base-v2)**

## 4. ChatBot
- [FinalModel](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/FinalModel)

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