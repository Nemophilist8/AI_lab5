import os
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from PIL import Image
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
from torchvision.transforms import transforms
import argparse
import CoattentionTransformerLayer
import torch.nn.functional as F



# 定义命令行参数
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_type', choices=['concat','gated', 'single-stream','double-stream'], default='concat',
                    help='Choose the model type for prediction')

args = parser.parse_args()
model_type = args.model_type

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

        input_ids = self.tokenizer.encode(description, return_tensors='pt', padding='max_length',
                                          max_length=self.max_length).squeeze(0)
        # print(img.shape, input_ids.shape)

        if len(input_ids) > 128:
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

class CustomBERTClassifier(nn.Module):
    def __init__(self, bert_model):
        super(CustomBERTClassifier, self).__init__()
        self.bert = bert_model

    def forward(self, inputs):
        outputs = self.bert(**inputs)
        return outputs.logits

class ConcatFusion(nn.Module):
    def __init__(self, x_dim = 2048, y_dim = 768, hidden_dim = 512, output_dim=3, dropout_prob=0.3):
        super(ConcatFusion, self).__init__()
        self.fc_x = nn.Linear(x_dim, hidden_dim)
        self.fc_y = nn.Linear(y_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(hidden_dim*2, output_dim)


    def forward(self, x, y):
        x = self.fc_x(x)
        y = self.fc_y(y)
        # Concatenate input tensors
        output = torch.cat((x, y), dim=1)

        output = self.dropout1(output)
        output = self.fc1(output)


        return x, y, output

class GatedFusion(nn.Module):
    def __init__(self, input_dim_x=512, input_dim_y=512, dim=512, output_dim=100, x_gate=True):
        super(GatedFusion, self).__init__()
        # 添加线性映射以使输入具有相同的维度
        self.fc_x = nn.Linear(input_dim_x, dim)
        self.fc_y = nn.Linear(input_dim_y, dim)
        self.fc_out = nn.Linear(dim, output_dim)
        self.x_gate = x_gate
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        # 添加线性映射以使输入具有相同的维度
        out_x = self.fc_x(x)
        out_y = self.fc_y(y)

        if self.x_gate:
            gate = self.sigmoid(out_x)
            output = self.fc_out(torch.mul(gate, out_y))
        else:
            gate = self.sigmoid(out_y)
            output = self.fc_out(torch.mul(out_x, gate))

        return out_x, out_y, output

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

class MultiModalCoTransformer(nn.Module):
    def __init__(self, input_size_img=2048, input_size_text=768, output_size=3, hidden_size=256, nhead=16):
        super(MultiModalCoTransformer, self).__init__()
        self.fc_img = nn.Linear(input_size_img, hidden_size)
        self.fc_text = nn.Linear(input_size_text, hidden_size)


        self.text_encoder = nn.TransformerEncoderLayer(
            d_model= hidden_size,
            nhead=nhead,
            # dropout=0.1
        )

        self.co_attention = CoattentionTransformerLayer.CoAttentionEncoderLayer(hidden_size,device)

        self.encoder1 = nn.TransformerEncoderLayer(
            d_model= hidden_size,
            nhead=nhead,
            # dropout=0.1
        )

        self.encoder2 = nn.TransformerEncoderLayer(
            d_model= hidden_size,
            nhead=nhead,
            # dropout=0.1
        )
        self.fc = nn.Linear(hidden_size*2, output_size)


    def forward(self, img_features, text_features):
        img_features = F.relu(self.fc_img(img_features))
        text_features = F.relu(self.fc_text(text_features))

        text_features = self.text_encoder(text_features)

        # Transpose for transformer input shape (sequence length, batch size, hidden size)
        img_features, _, text_features, _ = self.co_attention(img_features, text_features)

        img_features = self.encoder1(img_features)
        text_features = self.encoder2(text_features)

        output = torch.cat((img_features, text_features), dim=1)

        output = self.fc(output)


        return output


if model_type == 'concat':
    fusion_model = ConcatFusion()
    fusion_model.load_state_dict(torch.load('concat.pth'))
elif model_type == 'gated':
    fusion_model = GatedFusion(input_dim_x=2048, input_dim_y=768, dim=512, output_dim=3, x_gate=False)
    fusion_model.load_state_dict(torch.load('gated_y.pth'))
elif model_type == 'single-stream':
    fusion_model = MultiModalTransformer()
    fusion_model.load_state_dict(torch.load('single.pth'))
elif model_type == 'double-stream':
    fusion_model = MultiModalCoTransformer()
    fusion_model.load_state_dict(torch.load('double.pth'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = CustomDataset('数据', transform)

# Define the data loader
batch_size = 16
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Initialize ResNet50 img_model
img_model = resnet50(pretrained=False)
img_model.fc = nn.Linear(img_model.fc.in_features, 3)  # Assuming 3 classes for classification

# Load the saved img_model weights
model_weights_path = 'resnet50.pth'
img_model.load_state_dict(torch.load(model_weights_path))
img_model.fc = nn.Identity()

# 初始化相同的 BERT 模型
bert_model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=3)
text_model = CustomBERTClassifier(bert_model)
text_model.load_state_dict(torch.load('bert-base-multilingual-cased.pth'))
text_model = text_model.bert.bert

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_model = img_model.to(device)
text_model = text_model.to(device)
fusion_model = fusion_model.to(device)

# Make predictions
img_model.eval()
text_model.eval()
val_loss = 0.0
correct = 0
total = 0

with torch.no_grad():
    for imgs, input_ids, labels in data_loader:
        imgs, input_ids, labels = imgs.to(device), input_ids.to(device), labels.to(device)
        outputs_img = img_model(imgs)
        outputs_text = text_model(input_ids)

        last_hidden_states = outputs_text.last_hidden_state

        # 提取特征向量（CLS对应的隐藏状态）
        outputs_text = last_hidden_states[:, 0, :]

        # print(outputs_img.shape, outputs_text.shape)

        if model_type == 'concat' or model_type == 'gated':
            _, _, outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))
        elif model_type == 'single-stream' or model_type == 'double-stream':
            outputs = fusion_model(outputs_img.to(device), outputs_text.to(device))

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
accuracy = correct / total
print(f'Accuracy: {accuracy * 100:.2f}%')

