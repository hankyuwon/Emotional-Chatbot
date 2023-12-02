# <Model 1>
# Model : BertModel (Pretrained on skt/kobert-base-v1)
# Tokenizer : KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
# vocab : BERTvocab
# input : [Question]
# Inference : [Emotion]


# <Model 2>
# Model : KoGPT2 (Pretrained on skt/kogpt2-base-v2)
# Tokenizer : PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2", bos_token=BOS, eos_token=EOS, unk_token='<unk>', pad_token=PAD, mask_token=MASK)
# input : [Inferenced Emotion] + [Question]
# Inference : [Answer]
import torch

import torch
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel
import torch
from transformers import GPT2LMHeadModel
import model1
import os


os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0")

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

checkpoint=torch.load("../EmotionQ-Answer_Training/Custom_Data/model_EQ2A_Custom_Data_label2map_120.pth", map_location=device)
model.load_state_dict(checkpoint["model"])

if __name__ == '__main__':
    with torch.no_grad():
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            a = ""
            emo_token = model1.predict(q)
            # print(emo_token)
            while 1:
                # input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(emo_token + SENT + Q_TKN + q + SENT + '0' + A_TKN + a)).unsqueeze(dim=0) # 0 은 무슨 의미?
                input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(emo_token + SENT + Q_TKN + q + SENT + '0' + A_TKN + a)).unsqueeze(dim=0) # 0 은 무슨 의미?
                pred = model(input_ids.to(device))
                pred = pred.logits
                gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                if gen == EOS:
                    break
                a += gen.replace("▁", " ")
            print("Chatbot > {}".format(a.strip()))