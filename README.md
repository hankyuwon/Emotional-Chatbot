# Emotional-Chatbot

* [Dataset and Checkpoints](#1-dataset-and-checkpoints)
* [Usage](#2-usage)
  * [Preparing for Forced Attention](#preparing-for-forced-attention)
  * [Model](#model)
  * [Inference](#inference)
* [Setting](#setting)

## Installation
```
pip install gluonnlp pandas tqdm
pip install mxnet
pip install sentencepiece
pip install transformers
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
pip install mxnet-mkl==1.6.0 numpy==1.23.1
pip install pytorch-lightning
```

## Dataset
- [Data Preprocessing](https://github.com/hankyuwon/Emotional-Chatbot/blob/develop/Data_preprocessing)


## Checkpoint
 - KoBERT 
    - Download [**(Pretrained Weight)**](https://drive.google.com/drive/folders/1V4v0ppYLoDvwemRnVpd-0QCYnCnqDSsl?hl=ko) and Replace ```'./Question-Emotion_Training/save_model```
    - [README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/Question-Emotion_Training)

 - KoGPT2
    -  Download [**(Pretrained Weight)**](https://drive.google.com/drive/folders/13MgcxhXt_BPmEg9-LK1y8Af2gPoBrRI2?hl=ko) and Replace ```'./EmotionQ-Answer_Training/save_model```
    - [README.md](https://github.com/hankyuwon/Emotional-Chatbot/tree/develop/EmotionQ-Answer_Training)

## Preparing for Forced Attention
-  Modify the code within the GPT2Attention class as follows: **Issues** [#2](https://github.com/hankyuwon/Emotional-Chatbot/issues/2)

### Inference
```
# cd FinalModel
python model.py --checkpoint <KoGPT2 CHECKPOINT_FILE NAME>
```


### Inference Example
There is **Custom_chatbotdataset(Validation).csv**
|label|Question|Answer|
|---|---|---
|37(우울한)|친구들이 요즘 나를 따돌려. 내가 혼자라는 게 너무 슬퍼.|혼자 된 기분이 너무 슬프겠어요.|
|25(성가신)|다음번 흡연하러 나가기 전에 담배를 들고 나가라고 이야기해야 할 것 같아.|다음 번에는 후임에게 담배를 들고 나가라고 말하려 하시는군요.|
|31(악의적인)|병이 있어서 일을 할 수 없는데 그 마음을 몰라주니 너무 화가 나.|병 때문에 일하지 못하는 마음을 몰라줘서 화가 나시는군요. 어떻게 속상함을 조금이라도 해소할 수 있을까요?|

<br>

```
python model.py --checkpoint model_EQ2A_Custom_Data_60label2map_120

user > 친구들이 요즘 나를 따돌려. 내가 혼자라는 게 너무 슬퍼.
Chatbot > 따돌림을 당해내느라 많이 힘드시군요

user > 다음번 흡연하러 나가기 전에 담배를 들고 나가라고 이야기해야 할 것 같아.
Chatbot > 다음번에는 후임이 잊지 않고 담배를 챙겼으면 좋겠어요

user > 병이 있어서 일을 할 수 없는데 그 마음을 몰라주니 너무 화가 나
Chatbot > 현재의 감정을 나눌 수 있는 사람이 주위에 있었으면 좋겠네요
```

```
python model.py --checkpoint model_EQ2A_Custom_Data_60label2map_Forced_Attention_120

user > 친구들이 요즘 나를 따돌려. 내가 혼자라는 게 너무 슬퍼.
Chatbot > 혼자라는 게 슬프겠지만 분명 많이 힘내실 거예요

user > 다음번 흡연하러 나가기 전에 담배를 들고 나가라고 이야기해야 할 것 같아.
Chatbot > 다음번에는 후임이 잊지 않고 담배를 챙겼으면 좋겠어요

user > 병이 있어서 일을 할 수 없는데 그 마음을 몰라주니 너무 화가 나
Chatbot > 지금 상황에 화가 나는 감정을 나눌 수 있는 사람이 있었으면 좋겠어요
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