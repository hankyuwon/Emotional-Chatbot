# Emotional-Chatbot

* [Dataset and Checkpoints](#1-dataset-and-checkpoints)
* [Usage](#2-usage)
  * [Preparing for Forced Attention](#preparing-for-forced-attention)
  * [Model](#model)
  * [Inference](#inference)
* [Setting](#setting)


## 1. Dataset and Checkpoints
- [Data Preprocessing](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/Data_preprocessing)

 - KoBERT 
    - [**(Pretrained Weight)**](https://drive.google.com/drive/folders/1V4v0ppYLoDvwemRnVpd-0QCYnCnqDSsl?hl=ko)
    - Question-Emotion_Training [README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/Question-Emotion_Training)

 - KoGPT2
    -  [**(Pretrained Weight)**](https://drive.google.com/drive/folders/13MgcxhXt_BPmEg9-LK1y8Af2gPoBrRI2?hl=ko)
    - EmotionQ-Answer_Training [README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/EmotionQ-Answer_Training)

## 2. Usage
- [FinalModel](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/FinalModel)

### Preparing for Forced Attention
-  Modify the code within the GPT2Attention class as follows: **Issues** [#2](https://github.com/hankyuwon/Emotional-Chatbot/issues/2)

### Model
| Model | Data | Emotion | label | ForcedAttention |
|---|---|---|---|---|
| model_EQ2A_OriginalData_noEmotion_30 | OriginalData | N | - | N |
| model_EQ2A_OriginalData_60label2map_120 | OriginalData | Y | to Token | N |
| model_EQ2A_Custom_Data_60label2string_30 | CustomData | Y | to String | N |
| model_EQ2A_Custom_Data_60label2map_120 | CustomData | Y | to Token | N |
| model_EQ2A_Custom_Data_60label2map_Forced_Attention_120 | CustomData | Y | to Token | Y |

### Inference
```
# cd FinalModel
python model.py --checkpoint <CHECKPOINT_FILE>
```


### Inference Example
```
python model.py --checkpoint model_EQ2A_Custom_Data_60label2map_120

user > 입사 새내기지만 우리 팀을 잘 이끌어 주시는 팀장님이 있어서 감사함을 느껴.
Chatbot > 팀장님께 고마움을 표현할 방법이 있을까요

user > 그 분이 보기와는 달리 실제로는 인간미 넘치고 주어진 업무는 엄청 깔끔하게 하셔.
Chatbot > 이미지와는 다른 모습이 있군요

user > 내가 더 믿고 신뢰해야지. 나는 그분께 정말 인정받는 신입이 되고 싶어.
Chatbot > 신뢰하는 제일 큰 이유는 무엇인가요
```

```
python model.py --checkpoint model_EQ2A_Custom_Data_60label2map_Forced_Attention_120

user >
Chatbot >

user >
Chatbot >

user >
Chatbot >
```

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