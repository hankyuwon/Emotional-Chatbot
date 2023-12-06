Question-Emotion_Training
-
### Introduction
Predict **emotion** based on the user's Question

### Data Preparation
You can use the custom_chatbotdataset(Training).csv and custom_chatbotdataset(Validation).csv files in the './Question-Emotion_Training' folder. please refer to [Data_preprocessing README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/Data_preprocessing/README.md).

### Installation
```bash
pip install gluonnlp pandas tqdm
pip install mxnet
pip install sentencepiece
pip install transformers
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
```

### Inference Example
```
user > 입사 새내기지만 우리 팀을 잘 이끌어 주시는 팀장님이 있어서 감사함을 느껴.
>>> Predicted Emotion is "감사하는"

user > 그 분이 보기와는 달리 실제로는 인간미 넘치고 주어진 업무는 엄청 깔끔하게 하셔.
>>> Predicted Emotion is "감사하는"

user > 내가 더 믿고 신뢰해야지. 나는 그분께 정말 인정받는 신입이 되고 싶어.
>>> Predicted Emotion is "신뢰하는"
```

#### Notes:
 - Please place the [downloaded](https://drive.google.com/drive/u/0/folders/1V4v0ppYLoDvwemRnVpd-0QCYnCnqDSsl) 'QtEmodel120.pth' weights into the './Question-Emotion_Training/save_model' folder.