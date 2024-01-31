import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from torchvision.transforms import transforms
import argparse
import torch.optim as optim


# 定义命令行参数
parser = argparse.ArgumentParser(description='Predict using ResNet and BERT models.')
parser.add_argument('--model_type', choices=['img', 'text'], default='img',
                    help='Choose the img_model type for prediction (img or text)')

class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs.logits

# 初始化相同的 BERT 模型
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
text_model = CustomBERTClassifier(bert_model)

text_model.load_state_dict(torch.load('bert-base-multilingual-cased.pth'))

text_model = text_model.bert.bert

class CustomDataset(Dataset):
    def __init__(self, datafile, transform=None, max_length=128, hashtags=True):
        self.datafile = datafile
        self.transform = transform
        self.hashtags = hashtags
        self.imgs = []
        self.descriptions = []
        self.labels = []
        self.max_length = max_length

        self.readdata()

        self.tokenizer = BertTokenizer.from_pretrained('./bert-base-multilingual-cased')

        print(len(self.imgs), len(self.descriptions), len(self.labels))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        input_ids = self.tokenizer.encode(description, return_tensors='pt', padding='max_length', max_length=self.max_length).squeeze(0)
        # print(img.shape, input_ids.shape)

        if len(input_ids)>128:
            input_ids = input_ids[:128]


        return img, input_ids, label

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

class MultiModalTransformer(nn.Module):
    def __init__(self, input_size_img=2048, input_size_text=768, output_size=3, hidden_size=256, nhead=16):
        super(MultiModalTransformer, self).__init__()

        self.fc_img = nn.Linear(input_size_img, hidden_size)
        self.fc_text = nn.Linear(input_size_text, hidden_size)

        # Transformer layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model= hidden_size*2,
            nhead=nhead,
            # dropout=0.1
        )

        self.dropout = nn.Dropout(p=0.3)
        self.fc = nn.Linear(hidden_size*2, output_size)


    def forward(self, img_features, text_features):
        img_proj = self.fc_img(img_features)
        text_proj = self.fc_text(text_features)

        combined_features = torch.cat((img_proj, text_proj), dim=1)

        combined_features = self.transformer_layer(combined_features)

        combined_features = torch.squeeze(combined_features, dim=1)

        output = self.dropout(combined_features)
        output = self.fc(output)

        return output

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset('数据', transform)

validation_split = 0.2
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
train_indices, val_indices = indices[split:], indices[:split]

# Define the data loader
batch_size = 8

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# Initialize ResNet50 img_model
img_model = resnet50(pretrained=False)
img_model.fc = nn.Linear(img_model.fc.in_features, 3)  # Assuming 3 classes for classification

# Load the saved img_model weights
model_weights_path = 'resnet50.pth'
img_model.load_state_dict(torch.load(model_weights_path))
img_model.fc = nn.Identity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_model = img_model.to(device)
text_model = text_model.to(device)

fusion_model = MultiModalTransformer()
fusion_model.to(device)
# print(fusion_model)
# exit(0)

img_model.eval()
text_model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)
num_epochs = 5
best_val_loss = float('inf')  # Initialize with a large value

for epoch in range(num_epochs):
    fusion_model.train()
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_pbar:
        for imgs, input_ids, labels in train_pbar:
            # print('Gradient norm:', torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 1.0))

            # print(imgs.shape, input_ids.shape, labels.shape)
            imgs, input_ids, labels = imgs.to(device), input_ids.to(device), labels.to(device)

            outputs_text = text_model(input_ids)

            last_hidden_states = outputs_text.last_hidden_state

            # 提取特征向量（CLS对应的隐藏状态）
            outputs_text = last_hidden_states[:, 0, :]
            outputs_img = img_model(imgs)


            optimizer.zero_grad()
            outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))
            # print(outputs)
            # print(type(outputs))
            # print(outputs.shape,labels.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_pbar.set_postfix({'Loss': loss.item()})

    fusion_model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, input_ids, labels in val_loader:
            imgs, input_ids, labels = imgs.to(device), input_ids.to(device), labels.to(device)
            outputs_img = img_model(imgs)
            outputs_text = text_model(input_ids)

            last_hidden_states = outputs_text.last_hidden_state

            # 提取特征向量（CLS对应的隐藏状态）
            outputs_text = last_hidden_states[:, 0, :]

            # print(outputs_img.shape, outputs_text.shape)

            outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # print(outputs)
            _, predicted = outputs.max(1)
            # print(predicted)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss}, Accuracy: {accuracy}%')

    # Save the img_model if it has the best validation loss
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = 'single.pth'
        torch.save(fusion_model.state_dict(), best_model_path)
        print(f'Saved the best img_model with validation loss: {best_val_loss} to {best_model_path}')