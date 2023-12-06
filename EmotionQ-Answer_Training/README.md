EmotionQ-Answer_Training
-
### Introduction
1. User의 질문을 받아 질문의 **감정**을 **예측**한다.
2. **예측된 감정**과 User의 질문을 KoGPT의 입력으로 사용한다.

### Installation
```bash
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install pytorch-lightning
```

### Inference Example
```
user > 입사 새내기지만 우리 팀을 잘 이끌어 주시는 팀장님이 있어서 감사함을 느껴.
>>> Predicted Emotion is "감사하는"
Chatbot > 팀장님께 고마움을 표현할 방법이 있을까요


user > 그 분이 보기와는 달리 실제로는 인간미 넘치고 주어진 업무는 엄청 깔끔하게 하셔.
>>> Predicted Emotion is "감사하는"
Chatbot > 이미지와는 다른 모습이 있군요


user > 내가 더 믿고 신뢰해야지. 나는 그분께 정말 인정받는 신입이 되고 싶어.
>>> Predicted Emotion is "신뢰하는"
Chatbot > 신뢰하는 제일 큰 이유는 무엇인가요
```

#### Notes:
 - Please place the [downloaded](https://drive.google.com/drive/u/0/folders/13MgcxhXt_BPmEg9-LK1y8Af2gPoBrRI2) weights into the './EmotionQ-Answer_Training/save_model' folder.