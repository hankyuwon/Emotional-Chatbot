EmotionQ-Answer_Training
-
### Introduction
1. User의 질문을 받아 질문의 **감정**을 **예측**한다.
2. **예측된 감정**과 User의 질문을 KoGPT의 입력으로 사용한다.


### Model
| Model | Data | Emotion | label | ForcedAttention |
|---|---|---|---|---|
| model_EQ2A_**OriginalData**_**noEmotion**_30 | OriginalData | N | - | N |
| model_EQ2A_**OriginalData**_**60label2map**_120 | OriginalData | Y | to Token | N |
| model_EQ2A_**Custom_Data**_**60label2string**_30 | CustomData | Y | to String | N |
| model_EQ2A_**Custom_Data**_**60label2map**_120 | CustomData | Y | to Token | N |
| model_EQ2A_**Custom_Data**_**60label2map**_**Forced_Attention**_120 | CustomData | Y | to Token | Y |


### Proposed Methodology
1. Difference between **Original Data** vs **Custom Data**

2. Presence or absence of **Emotion Token**
   ```python
   # Q_TKN : <usr>
   # q : user chat
   # SENT : <unused1>
   # A_TKN : <sys>
   # a : Inferenced Answer(initial: "")

   # No Emotion Token mean chatbot does not learn emotions
   
   koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)

   # Emotion Token
   KoGPT2_TOKENIZER.encode(emo_token + SENT + Q_TKN + q + SENT + A_TKN + a)
    ```

3. Whether the label converted to a **string** or to a **token**
   ```python
   # to string mean
   class ChatbotDataset(Dataset):
    ...
    
    def __getitem__(self, idx):
        turn = self._data.iloc[idx]

        emotion = turn["label"] # An example of emotion : 1, 2, 3, ...
        emotion = str(emotion) # int -> string


   # to token mean

   emotion2Token_map = {
    9: '<unused10>', 11: '<unused11>', 2: '<unused12>', 12: '<unused13>', 13: '<unused14>', 15: '<unused15>',
    16: '<unused16>', 18: '<unused17>', ...
    }

   for i in range(len(Chatbot_Data['label'].nunique())):
      Chatbot_Data.replace({'label' :i}, emotion2Token_map[i], inplace = True)

   class ChatbotDataset(Dataset):
    ...

    def __getitem__(self, idx):
        turn = self._data.iloc[idx]

        emotion = turn["label"] # An example of emotion : '<unused10>', '<unused11>' ...
   ```