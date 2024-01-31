import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision.models import resnet50
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torchvision.transforms import transforms
import torch.optim as optim

class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, inputs):
        # 前向传播逻辑
        outputs = self.bert(**inputs)
        return outputs.logits

# 初始化相同的 BERT 模型
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
loaded_model = CustomBERTClassifier(bert_model)
loaded_model.load_state_dict(torch.load('bert-base-multilingual-cased.pth'))
bert_model = loaded_model.bert.bert

class CustomDataset(Dataset):
    def __init__(self, datafile, transform=None, max_length=128, hashtags=True):
        # 构造函数
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
        # 返回数据集大小
        return len(self.labels)

    def __getitem__(self, idx):
        # 获取数据集中的样本
        img = self.imgs[idx]
        description = self.descriptions[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        input_ids = self.tokenizer.encode(description, return_tensors='pt', padding='max_length', max_length=self.max_length).squeeze(0)

        if len(input_ids) > 128:
            input_ids = input_ids[:128]

        return img, input_ids, label

    def readdata(self):
        # 读取数据
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
                    description = description.replace('#', '')

                img = Image.open(img_path).convert('RGB')
                self.imgs.append(img)
                self.labels.append(label)
                self.descriptions.append(description)

    def remove_hashtags(self, sentence):
        # 去除文本中的hashtags
        cleaned_sentence = re.sub(r'\#\w+', '', sentence)
        return cleaned_sentence.strip()

    def remove_rt_mentions(self, sentence):
        # 去除文本中的RT mentions
        cleaned_sentence = re.sub(r'RT\s?@\w+:\s?', '', sentence)
        return cleaned_sentence.strip()

    def remove_at(self, sentence):
        # 去除文本中的@
        cleaned_sentence = re.sub(r'@\w+', '', sentence)
        return cleaned_sentence.strip()

    def remove_urls(self, sentence):
        # 去除文本中的URLs
        cleaned_sentence = re.sub(r'http[s]?://t\.co\w+', '', sentence)
        return cleaned_sentence.strip()

class ConcatFusion(nn.Module):
    def __init__(self, x_dim=2048, y_dim=768, hidden_dim=512, output_dim=3, dropout_prob=0.3):
        """
        :param x_dim: 第一个输入张量 `x` 的特征维度
        :param y_dim: 第二个输入张量 `y` 的特征维度
        :param hidden_dim: 融合过程中的隐藏层维度
        :param output_dim: 输出的维度，通常对应任务的类别数
        :param dropout_prob: Dropout 操作的概率。
        """
        super(ConcatFusion, self).__init__()
        self.fc_x = nn.Linear(x_dim, hidden_dim)
        self.fc_y = nn.Linear(y_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x, y):
        # 前向传播逻辑
        x = self.fc_x(x)
        y = self.fc_y(y)
        # 连接输入张量
        output = torch.cat((x, y), dim=1)

        output = self.dropout1(output)
        output = self.fc1(output)

        return x, y, output

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

# 定义数据加载器
batch_size = 16

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)

# 初始化 ResNet50 img_model
model = resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 3)  # 假设有3个分类

# 加载保存的 img_model 权重
model_weights_path = 'resnet50.pth'
model.load_state_dict(torch.load(model_weights_path))
model.fc = nn.Identity()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

fusion_model = ConcatFusion()
fusion_model.to(device)

model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001)
num_epochs = 5
best_val_loss = float('inf')  # 初始值设为一个较大的数

for epoch in range(num_epochs):
    fusion_model.train()
    with tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as train_pbar:
        for imgs, input_ids, labels in train_pbar:
            imgs, input_ids, labels = imgs.to(device), input_ids.to(device), labels.to(device)
            outputs_img = model(imgs)
            outputs_text = bert_model(input_ids)

            last_hidden_states = outputs_text.last_hidden_state

            # 提取特征向量（CLS对应的隐藏状态）
            outputs_text = last_hidden_states[:, 0, :]

            optimizer.zero_grad()
            _, _, outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))

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
            outputs_img = model(imgs)
            outputs_text = bert_model(input_ids)

            last_hidden_states = outputs_text.last_hidden_state

            # 提取特征向量（CLS对应的隐藏状态）
            outputs_text = last_hidden_states[:, 0, :]

            _, _, outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))

            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    accuracy = 100 * correct / total

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_val_loss}, Accuracy: {accuracy}%')

    # 如果具有最佳的验证损失，则保存 img_model
    if avg_val_loss < best_val_loss and accuracy > 70:
        best_val_loss = avg_val_loss
        best_model_path = 'concat.pth'
        torch.save(fusion_model.state_dict(), best_model_path)
        print(f'Saved the best img_model with validation loss: {best_val_loss} to {best_model_path}')
