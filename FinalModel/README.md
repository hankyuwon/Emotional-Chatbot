Final Model 1
- 

### Introduction

You can download CHECKPOINT 'model_EQ2A_Custom_Data_60label2map_120', 'model_EQ2A_Custom_Data_60label2string_30', 'model_EQ2A_OriginalData_noEmotion_30'

### Preparing for Forced Attention
1. Go to "**CONDA_ENV**/lib/**pythonx.x**/site-packages/transformers/models/gpt2/modeling_gpt2.py"
2. Modify the code within the GPT2Attention class as follows:

```python
class GPT2Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
            ...
            
            # Add below code
            self.forced_bias = nn.Parameter(torch.zeros(self.num_heads, 1, self.head_dim))

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            ...
            
            # Modify below code
            attn_output = torch.matmul(attn_weights, value) + self.forced_bias
            return attn_output, attn_weights
```

```python
attn_output.shape
>>> (batch_size, num_heads, max_len, (hidden_dims/num_heads))

# 40개의 토큰 중, 맨 앞(감정 토큰) 에 대한 Bias를 더해주기 위함
forced_bias.shape
>>> (num_heads, 1, (hidden_dims/num_heads))
```


### Inference
```
python model.py --checkpoint <CHECKPOINT_FILE>
```

#### Inference Example
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
python model.py --checkpoint model_EQ2A_Custom_Data_60label2string_30

```
```
python model.py --checkpoint model_EQ2A_OriginalData_noEmotion_30

```
```
python model.py --checkpoint

```