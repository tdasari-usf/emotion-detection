import os
import torch
import pandas as pd

from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast
from torch.utils.data import DataLoader
from transformers import DistilBertForSequenceClassification, AdamW
import torch.nn.functional as F

'''
c1 = 'wget -P ./ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_1.csv'
c2 = 'wget -P ./ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_2.csv'
c3 = 'wget -P ./ https://storage.googleapis.com/gresearch/goemotions/data/full_dataset/goemotions_3.csv'
os.system(c1)
os.system(c2)
os.system(c3)
'''

path_1 = './goemotions_1.csv'
d1 = pd.read_csv(path_1)

path_2 = './goemotions_2.csv'
d2 = pd.read_csv(path_2)


path_3 = './goemotions_3.csv'
d3 = pd.read_csv(path_3)
cols = ["text", "id", "admiration", "annoyance"]
df = d1[['id','text','admiration','annoyance']]
df_all = pd.concat([d1,d2,d3])

df_admiration = df_all[df_all.admiration == 1]
df_annoyance = df_all[df_all.annoyance == 1]

data_admiration = df_admiration.text.tolist()
data_annoyance = df_annoyance.text.tolist()


#assign label 1 to admiration, 0 to annoyance
labeled_texts = []
for text in data_admiration:
    labeled_texts.append((text, 1))
for text in data_annoyance:
    labeled_texts.append((text, 0))

train_texts, test_texts, train_labels, test_labels = train_test_split(
    [text[0] for text in labeled_texts], 
    [text[1] for text in labeled_texts], 
    test_size=0.1,                       
    random_state=42,                     
    stratify=[text[1] for text in labeled_texts] 
)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts,
                                                                    train_labels,
                                                                    test_size=0.1, 
                                                                    random_state = 42)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)
test_encodings = tokenizer(test_texts, truncation=True, padding=True)



class dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = dataset(train_encodings, train_labels)
val_dataset = dataset(val_encodings, val_labels)
test_dataset = dataset(test_encodings, test_labels)



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Training on device {device}...")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

for epoch in range(3):
  print(f"epoch - {epoch}")
  bt = 0
  for batch in train_loader:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs[0]
    if bt % 500 == 0:
    print(f"epoch - {epoch} batch - {bt} - loss - {loss.item()}")
    bt += 1
    loss.backward()
    optim.step()
    break

  print(f"end of epoch {epoch} - total batches - {bt}")





