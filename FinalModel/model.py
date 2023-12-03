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
import Q2E_model as Q2E_model
import os
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device("cuda:0")

Q_TKN = "<usr>"
A_TKN = "<sys>"
BOS = '</s>'
EOS = '</s>'
MASK = '<unused0>'
SENT = '<unused1>'
PAD = '<pad>'

parser = argparse.ArgumentParser(description='Emotion-Chatbot')

parser.add_argument('--checkpoint',
                    default = 'model_EQ2A_Custom_Data_label2map_120',
                    help = 'Please select the checkpoint. You can choose from: model_EQ2A_Custom_Data_label2map_120, model_EQ2A_Custom_Data_label2string_30, model_EQ2A_Custom_Data_Original_20, model_EQ2A_OriginalData_NoEmotion_30. '
)

args = parser.parse_args()

if __name__ == '__main__':
    
    koGPT2_TOKENIZER = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
            bos_token=BOS, eos_token=EOS, unk_token='<unk>',
            pad_token=PAD, mask_token=MASK) 
    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')

    model = model.to(device)
    
    checkpoint = torch.load("../EmotionQ-Answer_Training/save_model/{}.pth".format(args.checkpoint), map_location=device)
    model.load_state_dict(checkpoint["model"])
    
    with torch.no_grad():
        while 1:
            q = input("user > ").strip()
            if q == "quit":
                break
            a = ""
            emo_token, emo_str = Q2E_model.predict(q)
            # print('Predicted Emotion is ',emo_str)
            
            # Emotion Token을 사용하지 않은 GPT model
            if args.checkpoint in 'noEmotion':
                while 1:
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)

                    pred = model(input_ids.to(device))
                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("Chatbot > {}".format(a.strip()))
                
            # Emotion Token을 사용하는 GPT model
            else:
                while 1:
                    input_ids = torch.LongTensor(koGPT2_TOKENIZER.encode(emo_token + SENT + Q_TKN + q + SENT + A_TKN + a)).unsqueeze(dim=0)

                    pred = model(input_ids.to(device))
                    pred = pred.logits
                    gen = koGPT2_TOKENIZER.convert_ids_to_tokens(torch.argmax(pred, dim=-1).squeeze().cpu().numpy().tolist())[-1]
                    if gen == EOS:
                        break
                    a += gen.replace("▁", " ")
                print("Chatbot > {}".format(a.strip()))