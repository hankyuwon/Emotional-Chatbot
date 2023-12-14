
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook
import pandas as pd
from sklearn.model_selection import train_test_split
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel
# from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# GPU 가 있는 경우 지정
device = torch.device("cuda:0")

# GPU 가 없는 경우 지정
# device = torch.device("cpu")

emotion_mapping = {9: '노여워하는', 11: '느긋', 2: '걱정스러운', 12: '당혹스러운', 13: '당황', 15: '마비된', 16: '만족스러운', 18: '배신당한', 19: '버려진', 20: '부끄러운', 21: '분노', 22: '불안', 23: '비통한',
 24: '상처', 25: '성가신', 26: '스트레스 받는', 27: '슬픔', 28: '신뢰하는', 29: '신이 난', 30: '실망한', 31: '악의적인', 32: '안달하는', 33: '안도', 34: '억울한', 35: '열등감', 36: '염세적인', 37: '외로운', 38: '우울한',
 3: '고립된', 41: '좌절한', 55: '후회되는', 51: '혐오스러운', 50: '한심한', 39: '자신하는', 6: '기쁨', 48: '툴툴대는', 8: '남의 시선을 의식하는', 54: '회의적인', 42: '죄책감의', 52: '혼란스러운', 45: '초조한',
 56: '흥분', 46: '충격 받은', 47: '취약한', 49: '편안한', 17: '방어적인', 43: '질투하는', 14: '두려운', 10: '눈물이 나는', 44: '짜증내는', 40: '조심스러운', 7: '낙담한', 53: '환멸을 느끼는', 57: '희생된', 1: '감사하는', 5: '구역질 나는',
 4: '괴로워하는', 0: '가난한, 불우한'}

emotion2Token_map = {
    9: '<unused10>', 11: '<unused11>', 2: '<unused12>', 12: '<unused13>', 13: '<unused14>', 15: '<unused15>',
    16: '<unused16>', 18: '<unused17>', 19: '<unused18>', 20: '<unused19>', 21: '<unused20>', 22: '<unused21>',
    23: '<unused22>', 24: '<unused23>', 25: '<unused24>', 26: '<unused25>', 27: '<unused26>', 28: '<unused27>',
    29: '<unused28>', 30: '<unused29>', 31: '<unused30>', 32: '<unused31>', 33: '<unused32>', 34: '<unused33>',
    35: '<unused34>', 36: '<unused35>', 37: '<unused36>', 38: '<unused37>', 3: '<unused38>', 41: '<unused39>',
    55: '<unused40>', 51: '<unused41>', 50: '<unused42>', 39: '<unused43>', 6: '<unused44>', 48: '<unused45>',
    8: '<unused46>', 54: '<unused47>', 42: '<unused48>', 52: '<unused49>', 45: '<unused50>', 56: '<unused51>',
    46: '<unused52>', 47: '<unused53>', 49: '<unused54>', 17: '<unused55>', 43: '<unused56>', 14: '<unused57>',
    10: '<unused58>', 44: '<unused59>', 40: '<unused60>', 7: '<unused61>', 53: '<unused62>', 57: '<unused63>',
    1: '<unused64>', 5: '<unused65>', 4: '<unused66>', 0: '<unused67>'
}

# Model 1

tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')
tok = tokenizer.tokenize

# Setting parameters
max_len = 64
batch_size = 256
warmup_ratio = 0.1
num_epochs = 120
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len,
                 pad, pair):
   
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))
         
    def __len__(self):
        return (len(self.labels))

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=58,
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
# Prepare optimizer and schedule (linear warmup and decay)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss().to(device)

# checkpoint=torch.load("../Question-Emotion_Training/save_model/QtEmodel120.pth", map_location=device)
checkpoint = torch.load("../Question-Emotion_Training/save_model/QtEmodel120.pth", map_location=device)
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])


def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]
    another_test = BERTDataset(dataset_another, 0, 1, tok, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size = batch_size, num_workers = 4)

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)
        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            test_eval.append([np.argmax(logits)])

    # print(">> 입력하신 내용의 감정은 '",emotion_mapping[test_eval[0][0]],"' 입니다.")
    # print(">> 입력하신 내용의 감정 라벨은 '",test_eval[0][0],"' 입니다.")
    # print(">> 입력하신 내용의 토큰은 " ,emotion2Token_map.get(test_eval[0][0]), " 입니다.")
    return emotion2Token_map.get(test_eval[0][0]), emotion_mapping[test_eval[0][0]]