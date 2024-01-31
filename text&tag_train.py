import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import accuracy_score
import re

datafile = '数据'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomDataset(Dataset):
    def __init__(self, datafile, transform=None, hashtags=False, max_length=128):
        self.datafile = datafile
        self.transform = transform
        self.hashtags = hashtags
        self.imgs = []
        self.descriptions = []
        self.labels = []
        self.max_length = max_length

        self.readdata()

        print(len(self.imgs), len(self.descriptions), len(self.labels))

        # Load BERT tokenizer and img_model
        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('./bert-base-multilingual-cased')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]

        # Tokenize description using BERT tokenizer
        inputs = self.tokenizer(description, return_tensors="pt", truncation=True, padding='max_length', max_length=self.max_length)
        # No need to move inputs to device in __getitem__

        # Forward pass through BERT img_model
        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        # Use the [CLS] token representation as the sentence representation
        sentence_representation = outputs.last_hidden_state[:, 0, :]

        # print(inputs['input_ids'].squeeze().shape,inputs['attention_mask'].squeeze().shape)

        return inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), label

    def remove_hashtags(self, sentence):
        cleaned_sentence = re.sub(r'\#\w+', '', sentence)
        return cleaned_sentence.strip()

    def remove_rt_mentions(self, sentence):
        cleaned_sentence = re.sub(r'RT\s?@\w+:\s?', '', sentence)
        return cleaned_sentence.strip()

    def remove_at(self, sentence):
        cleaned_sentence = re.sub(r'@\w+', '', sentence)
        return cleaned_sentence.strip()

    def remove_urls(self, sentence):
        cleaned_sentence = re.sub(r'http[s]?://t\.co\w+', '', sentence)
        return cleaned_sentence.strip()

    def readdata(self):
        with open(os.path.join(self.datafile, 'train.txt'), 'r', encoding='utf-8') as fp:
            lines = fp.readlines()
            lines = lines[1:]
            lines = [i[:-1] for i in lines]
            for line in lines:
                t = line.split(',')
                guid = t[0]
                label = t[1]
                if label == 'positive':
                    label = 0
                elif label == 'neutral':
                    label = 1
                else:
                    label = 2

                img_path = os.path.join(self.datafile, 'data', guid + '.jpg')
                txt_path = os.path.join(self.datafile, 'data', guid + '.txt')

                description = ''
                try:
                    with open(txt_path, 'r', encoding='gbk') as fp1:
                        description = fp1.read()[:-1]
                except:
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as fp1:
                            description = fp1.read()[:-1]
                    except:
                        print(txt_path)
                        continue

                description = self.remove_rt_mentions(description)
                description = self.remove_at(description)
                description = self.remove_urls(description)
                description = description.lower()

                if not self.hashtags:
                    description = self.remove_hashtags(description)
                else:
                    description = description.replace('#','')

                img = Image.open(img_path).convert('RGB')
                self.imgs.append(img)
                self.labels.append(label)
                self.descriptions.append(description)

class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs.logits

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = CustomDataset(datafile, transform)


# Split dataset into training and validation sets
validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Create data loaders without multi-threading
batch_size = 16

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)


tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
model = CustomBERTClassifier(bert_model)
model = model.to(device)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, save_path='best_model.pth'):
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_pbar:
            for batch in train_pbar:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                logits = model({'input_ids': input_ids, 'attention_mask': attention_mask})
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                train_pbar.set_postfix({'Loss': loss.item()})

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                logits = model({'input_ids': input_ids, 'attention_mask': attention_mask})
                loss = criterion(logits, labels)
                val_loss += loss.item()

                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss}, Accuracy: {accuracy}%')

        # Save the img_model if accuracy improves
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), save_path)
            print(f'Saved the model with best accuracy: {best_accuracy}%')

# 使用时指定保存路径
save_path = 'bert-base-multilingual-cased_tag.pth'


criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-5)

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, save_path=save_path)