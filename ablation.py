import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW
from torchvision.transforms import transforms
import argparse

class CustomDataset(Dataset):
    def __init__(self, datafile, transform=None, max_length=128):
        self.datafile = datafile
        self.transform = transform
        self.imgs = []
        self.descriptions = []
        self.labels = []
        self.max_length = max_length

        self.readdata()

        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-cased')
        self.bert_model = BertModel.from_pretrained('./bert-base-multilingual-cased')


        print(len(self.imgs), len(self.descriptions), len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        # Tokenize description using BERT tokenizer
        inputs = self.tokenizer(description, return_tensors="pt", truncation=True, padding='max_length', max_length=self.max_length)
        # No need to move inputs to device in __getitem__

        # Forward pass through BERT img_model
        with torch.no_grad():
            outputs = self.bert_model(**inputs)


        return img, inputs['input_ids'].squeeze(), inputs['attention_mask'].squeeze(), label

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
                        description=fp1.read()[:-1]
                except:
                    try:
                        with open(txt_path, 'r', encoding='utf-8') as fp1:
                            description=fp1.read()[:-1]
                    except:
                        print(txt_path)
                        continue

                img = Image.open(img_path).convert('RGB')
                self.imgs.append(img)
                self.labels.append(label)
                self.descriptions.append(description)

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

dataset = CustomDataset('数据', transform)

# Define the data loader
batch_size = 32
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize ResNet50 img_model
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # Assuming 3 classes for classification

# Load the saved img_model weights
model_weights_path = 'resnet50.pth'
model.load_state_dict(torch.load(model_weights_path))

# 初始化相同的 BERT 模型
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
loaded_model = CustomBERTClassifier(bert_model)
loaded_model.load_state_dict(torch.load('bert-base-multilingual-cased.pth'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
loaded_model = loaded_model.to(device)

# Make predictions
model.eval()
loaded_model.eval()
all_predictions_img = []
all_predictions_text = []
all_labels = []

with torch.no_grad():
    for imgs, input_ids, attention_mask, labels in tqdm(data_loader, desc='Predicting', unit='batch'):
        imgs, input_ids, attention_mask, labels = imgs.to(device), input_ids.to(device), attention_mask.to(device), labels.to(device)
        outputs = model(imgs)
        _, predicted = outputs.max(1)
        # print('img',predicted)
        all_predictions_img.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        logits = loaded_model({'input_ids': input_ids, 'attention_mask': attention_mask})
        _, predicted = logits.max(1)
        # print('text',predicted)

        all_predictions_text.extend(predicted.cpu().numpy())
# Calculate and print overall accuracy
accuracy_img = accuracy_score(all_labels, all_predictions_img)
print(f'Img Accuracy: {accuracy_img * 100:.2f}%')
accuracy_text = accuracy_score(all_labels, all_predictions_text)
print(f'Text Accuracy: {accuracy_text * 100:.2f}%')
