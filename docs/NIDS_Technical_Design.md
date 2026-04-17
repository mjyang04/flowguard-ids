# Lightweight CNN-BiLSTM-SE IDS with Explainability and Cross-Dataset Generalization）
## 技术设计文档与实施计划

**文档版本**: v1.0
**创建日期**: 2026-03-18
**项目**: Lightweight CNN-BiLSTM-SE IDS with Explainability and Cross-Dataset Generalization
**目标读者**: Codex

---

## 目录

1. [系统总体架构](#1-系统总体架构)
2. [数据处理流程](#2-数据处理流程)
3. [模型设计](#3-模型设计)
4. [SHAP特征选择流程](#4-shap特征选择流程)
5. [训练流程设计](#5-训练流程设计)
6. [Cross-dataset实验设计](#6-cross-dataset实验设计)
7. [评估指标体系](#7-评估指标体系)
8. [项目代码结构设计](#8-项目代码结构设计)
9. [模块接口定义](#9-模块接口定义)
10. [开发优先级与实施步骤](#10-开发优先级与实施步骤)
11. [RTX3060优化策略](#11-rtx3060优化策略)
12. [潜在风险与解决方案](#12-潜在风险与解决方案)

---

## 1. 系统总体架构

### 1.1 模块划分

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         NIDS 系统架构图                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  数据采集层   │    │  数据预处理层  │    │   模型训练层  │             │
│  │              │    │              │    │              │             │
│  │ CICIDS2017   │───▶│ 特征对齐     │───▶│ CNN-BiLSTM  │             │
│  │ UNSW-NB15    │    │ 数据清洗     │    │ CNN-BiLSTM-SE│             │
│  │              │    │ 归一化编码   │    │ +SHAP        │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│         │                   │                   │                       │
│         │                   │                   │                       │
│         ▼                   ▼                   ▼                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  特征工程层   │    │  模型推理层   │    │  评估分析层   │             │
│  │              │    │              │    │              │             │
│  │ SHAP分析     │    │ 实时检测     │    │ 指标计算     │             │
│  │ 特征选择     │    │ 批量推理     │    │ 可视化报告   │             │
│  │ Top-K筛选    │    │ 模型部署     │    │ 性能对比     │             │
│  └──────────────┘    └──────────────┘    └──────────────┘             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 核心模块说明

| 模块 | 职责 | 关键技术 |
|------|------|----------|
| **数据采集** | 加载原始CSV/NetFlow数据 | pandas, numpy |
| **数据预处理** | 清洗、对齐、编码、归一化 | sklearn, imblearn |
| **特征工程** | SHAP分析、特征选择 | shap, feature importance |
| **模型训练** | 训练、验证、early stopping | PyTorch 2.5.1 CUDA124 , AMP |
| **模型推理** | 批量/实时推理 | TorchScript, ONNX |
| **评估分析** | 指标计算、可视化 | sklearn, matplotlib |

---

## 2. 数据处理流程

### 2.1 数据集说明

| 数据集 | 原始特征数 | 样本量(估) | 攻击类型数 | 格式 |
|--------|-----------|-----------|------------|------|
| **CICIDS2017** | ~80 | 约280万 | 14种 | CSVISCX |
| **UNSW-NB15** | ~49 | 约200万 | 9种 | CSV |

### 2.2 特征对齐策略（重点）

**推荐方案：公共特征子集（Intersection Features）**

由于两个数据集特征空间不一致，采用**交集特征策略**：

```python
# 伪代码：特征对齐流程
def align_features(cicids_features: List[str], unsw_features: List[str]) -> List[str]:
    """
    1. 标准化特征名称（去除空格、转小写）
    2. 计算交集
    3. 手动映射相似特征（如flow_bytes/s ↔ sbytes）
    """
    common = set(cicids_features) & set(unsw_features)
    # 手动映射表
    manual_mapping = {
        'duration': 'dur',
        'total_bytes': 'sbytes',
        'packet_count': 'spkts',
        # ... 根据实际数据调整
    }
    return list(common | set(manual_mapping.keys()))
```

**统一特征空间（推荐使用NetFlow格式）：**

```
┌────────────────────────────────────────────────────────────────┐
│                    统一NetFlow格式特征                          │
├────────────────────────────────────────────────────────────────┤
│ 基础特征 (10):                                                   │
│   - src_port, dst_port, protocol,                              │
│   - total_fwd_packets, total_bwd_packets,                      │
│   - fwd_packet_len_mean, bwd_packet_len_mean,                  │
│   - flow_duration, flow_bytes_per_sec, flow_packets_per_sec   │
│                                                                │
│ 统计特征 (20+):                                                  │
│   - fwd_packet_len_min/max/std, bwd_packet_len_min/max/std,    │
│   - packet_len_mean/std/min/max,                              │
│   - flow_iat_mean/std/min/max,                                │
│   - fwd_iat_total/mean/std/min/max,                           │
│   - bwd_iat_total/mean/std/min/max                            │
│                                                                │
│ 派生特征 (10):                                                   │
│   - fwd_header_len_total, bwd_header_len_total,               │
│   - flags_count, active_mean/std/min/max,                     │
│   - idle_mean/std/min/max                                      │
└────────────────────────────────────────────────────────────────┘
```

### 2.3 数据清洗

```python
# 步骤1: 去除NaN/Inf
def clean_nan_inf(df: pd.DataFrame) -> pd.DataFrame:
    # 替换Inf为NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # 填充NaN为0（流量数据常用处理方式）
    df = df.fillna(0)
    return df

# 步骤2: 处理异常值（极大流量值）
def handle_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 1e6) -> pd.DataFrame:
    for col in columns:
        # 超过阈值的替换为阈值
        df[col] = df[col].clip(upper=threshold)
    return df

# 步骤3: 删除无效特征
def remove_invalid_features(df: pd.DataFrame) -> pd.DataFrame:
    # 删除常量特征（所有值相同）
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=constant_cols)

    # 删除重复特征
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > 0.95)]
    df = df.drop(columns=to_drop)
    return df
```

### 2.4 类别标签处理

**推荐：二分类（Benign vs Attack）**

| 原始标签 | 映射后标签 | 说明 |
|----------|-----------|------|
| BENIGN | 0 (Benign) | 正常流量 |
| DDoS, DoS* | 1 (Attack) | 拒绝服务攻击 |
| Bot, PortScan | 1 (Attack) | 僵尸网络/端口扫描 |
| Web Attack*, FTP-Patator, SSH-Patator | 1 (Attack) | 暴力破解/入侵 |
| Infiltration, Heartbleed | 1 (Attack) | 高级攻击 |

```python
# Label mapping伪代码
LABEL_MAPPING_BINARY = {
    'BENIGN': 0,
    'Bot': 1, 'DDoS': 1, 'DoS GoldenEye': 1, 'DoS Hulk': 1,
    'DoS Slowhttptest': 1, 'DoS slowloris': 1, 'FTP-Patator': 1,
    'Heartbleed': 1, 'Infiltration': 1, 'PortScan': 1,
    'SSH-Patator': 1, 'Web Attack - Brute Force': 1,
    'Web Attack - Sql Injection': 1, 'Web Attack - XSS': 1
}

# 或多分类（15类，保留攻击类型细粒度）
LABEL_MAPPING_MULTI = {
    'BENIGN': 0, 'Bot': 1, 'DDoS': 2, 'DoS GoldenEye': 3,
    'DoS Hulk': 4, 'DoS Slowhttptest': 5, 'DoS slowloris': 6,
    'FTP-Patator': 7, 'Heartbleed': 8, 'Infiltration': 9,
    'PortScan': 10, 'SSH-Patator': 11, 'Web Attack - Brute Force': 12,
    'Web Attack - Sql Injection': 13, 'Web Attack - XSS': 14
}
```

### 2.5 类别不平衡处理

**方案1：Class Weights（推荐，轻量级）**

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def compute_class_weights(y_train: np.ndarray) -> np.ndarray:
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    return weights  # 返回 [weight_0, weight_1, ...]
```

**方案2：SMOTE过采样（可选）**

```python
from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train, strategy='auto'):
    smote = SMOTE(sampling_strategy=strategy, random_state=42)
    return smote.fit_resample(X_train, y_train)
```

**方案3：Undersampling（可选）**

```python
from imblearn.under_sampling import RandomUnderSampler

def apply_undersampling(X_train, y_train):
    rus = RandomUnderSampler(random_state=42)
    return rus.fit_resample(X_train, y_train)
```

### 2.6 特征编码

**Categorical特征处理：**

```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 方案A: Label Encoding（特征维度小时使用）
def label_encode(df, cat_columns):
    encoders = {}
    for col in cat_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

# 方案B: One-Hot Encoding（推荐）
def onehot_encode(df, cat_columns):
    # sklearn ColumnTransformer会自动处理
    pass

# CICIDS2017分类特征（示例）
CATEGORICAL_FEATURES = ['Protocol', 'Service', 'Flag']
# UNSW-NB15分类特征（示例）
UNSW_CATEGORICAL_FEATURES = ['proto', 'state', 'service']
```

### 2.7 特征归一化

**关键：Scaler只在训练集fit！**

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

# 推荐：MinMaxScaler（网络流量特征通常有边界）
scaler = MinMaxScaler()

# 训练集：fit + transform
X_train_scaled = scaler.fit_transform(X_train)

# 验证/测试集：仅transform（防止数据泄露！）
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 保存scaler供推理使用
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
```

### 2.8 SHAP前的数据准备

```python
def prepare_shap_samples(X_train: np.ndarray, n_samples: int = 2000,
                        random_state: int = 42) -> np.ndarray:
    """
    从训练集中采样用于SHAP分析
    """
    np.random.seed(random_state)
    indices = np.random.choice(len(X_train), size=n_samples, replace=False)
    return X_train[indices]

# 保存SHAP样本
shap_samples = prepare_shap_samples(X_train_scaled, n_samples=2000)
np.save('data/shap_samples.npy', shap_samples)
```

### 2.9 最终输入格式

```python
# 模型输入格式
INPUT_SHAPE = (batch_size, feature_dim)
# 示例: (128, 55)  # 55个特征

# 对于CNN，需要reshape为序列格式
# (batch, 1, seq_len) = (batch, 1, 55)
x = inputs.unsqueeze(1)
```

### 2.10 数据加载模块设计

```python
from torch.utils.data import Dataset, DataLoader
import numpy as np

class NIDSDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# DataLoader工厂函数
def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                      batch_size=128, num_workers=0):
    train_ds = NIDSDataset(X_train, y_train)
    val_ds = NIDSDataset(X_val, y_val)
    test_ds = NIDSDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                           shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
```

---

## 3. 模型设计

### 3.1 模型架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    CNN-BiLSTM-SE 模型架构                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: (batch, feature_dim) = (128, 55)                       │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     ConvBlock 1                      │                      │
│  │  Conv1d(1, 64, k=3) → BN → ReLU     │                      │
│  │  → MaxPool(2) → Dropout(0.3)        │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     SE Block (Squeeze-Excitation)    │                      │
│  │  GlobalAvgPool → FC(64→4) → FC(4→64)│                      │
│  │  → Sigmoid → Scale                   │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     ConvBlock 2                      │                      │
│  │  Conv1d(64, 128, k=3) → BN → ReLU    │                      │
│  │  → MaxPool(2) → Dropout(0.3)         │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     BiLSTM Layer                     │                      │
│  │  hidden_size=128, num_layers=2,      │                      │
│  │  bidirectional=True                  │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     Attention Pooling (可选)         │                      │
│  │  Additive Attention                  │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  ┌──────────────────────────────────────┐                      │
│  │     Classifier                       │                      │
│  │  Dropout(0.3) → Linear(256→128)     │                      │
│  │  → ReLU → Dropout(0.3)              │                      │
│  │  → Linear(128→num_classes)           │                      │
│  └──────────────────────────────────────┘                      │
│                    │                                            │
│                    ▼                                            │
│  Output: (batch, num_classes) 或 (batch, 1)                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 详细参数配置

**推荐参数（适配RTX3060）：**

| 组件 | 参数 | 推荐值 | 说明 |
|------|------|--------|------|
| **输入** | input_dim | 55 | 统一特征空间维度 |
| **输出** | num_classes | 2 | 二分类（可改为15） |
| **ConvBlock1** | out_channels | 64 | 第一层卷积通道 |
| | kernel_size | 3 | 卷积核大小 |
| | pool_size | 2 | 池化大小 |
| | dropout | 0.3 | Dropout率 |
| **SE Block** | reduction | 16 | 压缩比 |
| **ConvBlock2** | out_channels | 128 | 第二层卷积通道 |
| | kernel_size | 3 | |
| | pool_size | 2 | |
| | dropout | 0.3 | |
| **BiLSTM** | hidden_size | 128 | LSTM隐藏层 |
| | num_layers | 2 | 层数 |
| | bidirectional | True | 双向 |
| | dropout | 0.3 | 层间dropout |
| **Classifier** | hidden_dim | 128 | 全连接层 |

**模型参数量估算：**

```
Conv1: 1*64*3 + 64 = 256
BN1: 64*4 = 256
SE1: 64*64/16 + 64*64/16 = 512
Conv2: 64*128*3 + 128 = 24,704
BN2: 128*4 = 512
SE2: 128*128/16 + 128*128/16 = 2048
LSTM: 4*(128*128 + 128*2)*2 = 132,096
FC1: 256*128 + 128 = 32,896
FC2: 128*2 + 2 = 258
------------------------------------------------
总计: ~200K 参数（轻量化设计）
```

### 3.3 PyTorch实现

```python
import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """卷积块：Conv -> BN -> ReLU -> Pool -> Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 pool_size=2, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                             padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class SqueezeExcitation(nn.Module):
    """SE模块：通道注意力机制"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        squeeze = x.mean(dim=-1)  # 全局平均池化
        excitation = torch.relu(self.fc1(squeeze))
        excitation = torch.sigmoid(self.fc2(excitation))
        scale = excitation.unsqueeze(-1)
        return x * scale


class AttentionPooling(nn.Module):
    """注意力池化"""
    def __init__(self, hidden_size):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        # x: (seq_len, batch, hidden)
        attn_scores = self.score(x)
        attn_weights = torch.softmax(attn_scores, dim=0)
        weighted = (attn_weights * x).sum(dim=0)
        return weighted


class CNNBiLSTMSE(nn.Module):
    """CNN-BiLSTM-SE模型"""
    def __init__(self, input_dim, num_classes,
                 conv_channels=[64, 128],
                 conv_kernel_sizes=[3, 3],
                 conv_pool_sizes=[2, 2],
                 lstm_hidden_size=128,
                 lstm_num_layers=2,
                 dropout=0.3,
                 bidirectional=True,
                 use_attention=False,
                 use_se=True):
        super().__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes

        # CNN特征提取器
        layers = []
        in_channels = 1
        seq_len = input_dim
        for i, (out_channels, kernel_size, pool_size) in enumerate(
            zip(conv_channels, conv_kernel_sizes, conv_pool_sizes)):
            layers.append(ConvBlock(in_channels, out_channels, kernel_size,
                                    pool_size, dropout))
            # SE模块
            if use_se:
                layers.append(SqueezeExcitation(out_channels))
            in_channels = out_channels
            seq_len = seq_len // pool_size
        self.feature_extractor = nn.Sequential(*layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=False,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_output_dim = lstm_hidden_size * (2 if bidirectional else 1)

        # 注意力池化（可选）
        self.attention = AttentionPooling(lstm_output_dim) if use_attention else None

        # 分类器
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, lstm_output_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(self, inputs):
        # inputs: (batch, features)
        x = inputs.unsqueeze(1)  # (batch, 1, seq_len)

        # CNN特征提取
        x = self.feature_extractor(x)

        # 转换为LSTM输入格式
        x = x.permute(2, 0, 1)  # (seq_len, batch, channels)

        # BiLSTM
        lstm_out, _ = self.lstm(x)

        # 池化
        if self.attention is not None:
            pooled = self.attention(lstm_out)
        else:
            pooled = lstm_out.mean(dim=0)

        # 分类
        logits = self.classifier(pooled)
        return logits.squeeze(-1) if self.num_classes == 2 else logits
```

---

## 4. SHAP特征选择流程

### 4.1 整体流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      SHAP特征选择流程                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. 准备SHAP样本                                                 │
│     └─ 从训练集采样2000条（分层采样，保证类别平衡）               │
│                                                                 │
│     2. 计算SHAP值                                                │
│        └─ 使用DeepSHAP或KernelSHAP                              │
│        └─ background数据集100条                                  │
│                                                                 │
│        3. 聚合特征重要性                                         │
│           └─ 对每个样本的SHAP值取绝对值                           │
│           └─ 对所有样本求均值，得到特征重要性排名                  │
│                                                                 │
│           4. 筛选Top-K特征                                       │
│              └─ 选择Top-K（如K=30）                              │
│              └─ 或选择累计重要性>90%的特征                       │
│                                                                 │
│              5. 重新训练模型                                      │
│                 └─ 使用筛选后的特征子集                          │
│                 └─ 对比性能差异                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 采样策略

```python
import numpy as np

def sample_for_shap(X_train: np.ndarray, y_train: np.ndarray,
                   n_samples: int = 2000, random_state: int = 42):
    """
    分层采样：确保各类别样本比例一致
    """
    np.random.seed(random_state)

    # 统计各类别数量
    classes, counts = np.unique(y_train, return_counts=True)
    n_classes = len(classes)

    # 计算每类采样数量
    samples_per_class = n_samples // n_classes

    indices = []
    for cls in classes:
        cls_indices = np.where(y_train == cls)[0]
        n_select = min(samples_per_class, len(cls_indices))
        selected = np.random.choice(cls_indices, size=n_select, replace=False)
        indices.extend(selected)

    # 补齐剩余样本
    if len(indices) < n_samples:
        remaining = set(range(len(X_train))) - set(indices)
        additional = np.random.choice(list(remaining),
                                      size=n_samples - len(indices),
                                      replace=False)
        indices.extend(additional)

    return X_train[indices], y_train[indices]
```

### 4.3 SHAP计算

```python
import shap

def compute_shap_values(model, X_samples, background_size=100):
    """
    计算SHAP值
    """
    # 创建background数据集（使用k-means压缩）
    background = shap.kmeans(X_samples[:background_size], 50)

    # 创建explainer
    explainer = shap.DeepExplainer(model,
                                   torch.from_numpy(X_samples[:background_size]).float())

    # 计算SHAP值
    shap_values = explainer.shap_values(torch.from_numpy(X_samples).float())

    return shap_values

def compute_feature_importance(shap_values):
    """
    计算特征重要性（绝对值均值）
    """
    # shap_values: (n_samples, n_features) 或 (n_samples, n_classes, n_features)
    if isinstance(shap_values, list):
        # 多分类：取各类的平均值
        importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        importance = np.abs(shap_values).mean(axis=0)

    return importance
```

### 4.4 Top-K筛选

```python
def select_top_k_features(feature_names, importance, k=30):
    """
    选择Top-K特征
    """
    # 排序
    sorted_idx = np.argsort(importance)[::-1]
    top_k_idx = sorted_idx[:k]

    selected_features = [feature_names[i] for i in top_k_idx]
    selected_importance = importance[top_k_idx]

    return selected_features, selected_importance, top_k_idx


def select_by_cumulative_importance(feature_names, importance, threshold=0.9):
    """
    选择累计重要性达到阈值的特征
    """
    sorted_idx = np.argsort(importance)[::-1]
    total_importance = importance.sum()

    cumulative = 0
    selected_idx = []
    for idx in sorted_idx:
        selected_idx.append(idx)
        cumulative += importance[idx]
        if cumulative / total_importance >= threshold:
            break

    selected_features = [feature_names[i] for i in selected_idx]
    return selected_features, selected_idx
```

### 4.5 保存和使用

```python
import json

def save_feature_selection_results(output_dir, feature_names, selected_features,
                                   importance, selected_idx):
    """保存特征选择结果"""
    results = {
        'all_features': feature_names,
        'selected_features': selected_features,
        'selected_indices': selected_idx.tolist(),
        'importance': importance.tolist(),
        'top_k': len(selected_features)
    }

    with open(f'{output_dir}/feature_selection.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 保存筛选后的数据处理管道
    np.save(f'{output_dir}/selected_feature_indices.npy', selected_idx)
```

---

## 5. 训练流程设计

### 5.1 训练配置参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **Batch Size** | 128 | RTX3060推荐 |
| **Epochs** | 30 | 最大轮数，配合early stopping |
| **Learning Rate** | 1e-3 | 初始学习率 |
| **Weight Decay** | 1e-4 | L2正则化 |
| **Optimizer** | Adam | 稳定收敛 |
| **Scheduler** | ReduceLROnPlateau | 监控验证指标 |
| **Scheduler Factor** | 0.5 | 学习率衰减因子 |
| **Scheduler Patience** | 2 | 等待epoch数 |
| **Min LR** | 1e-6 | 最小学习率 |
| **Early Stopping Patience** | 5 | 早停耐心值 |
| **Early Stopping Delta** | 1e-4 | 最小改善 |
| **Gradient Clip** | 1.0 | 梯度裁剪 |
| **AMP** | True | 混合精度训练 |

### 5.2 训练循环

```python
def train_model(model, train_loader, val_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(),
                                  lr=config.learning_rate,
                                  weight_decay=config.weight_decay)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.min_learning_rate
    )

    # 损失函数（带类别权重）
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))

    # AMP
    scaler = torch.amp.GradScaler('cuda') if config.amp else None

    # Early stopping
    best_metric = -float('inf')
    patience_counter = 0

    for epoch in range(1, config.num_epochs + 1):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion,
                                optimizer, device, scaler)

        # 验证
        val_metrics = evaluate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step(val_metrics['avg_attack_recall'])

        # Early stopping
        current_metric = val_metrics[config.selection_metric]
        if current_metric > best_metric + config.early_stopping_delta:
            best_metric = current_metric
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

### 5.3 训练步骤详解

```python
def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0

    for features, labels in dataloader:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(features)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    metrics = compute_nids_metrics(all_labels, all_preds)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics
```

---

## 6. Cross-dataset实验设计

### 6.1 实验设置

```
┌─────────────────────────────────────────────────────────────────┐
│                    Cross-Dataset实验设计                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  实验1: 同数据集验证                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CICIDS2017训练 ──▶ CICIDS2017测试                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  实验2: 跨数据集泛化（主实验）                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  CICIDS2017训练 ──▶ UNSW-NB15测试                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  实验3: 反向泛化                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  UNSW-NB15训练 ──▶ CICIDS2017测试                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 数据集对齐处理

```python
def prepare_cross_dataset(cicids_path, unsw_path, feature_config):
    """
    准备跨数据集训练/测试
    """
    # 加载CICIDS2017
    cicids_df = load_cicids(cicids_path)

    # 加载UNSW-NB15
    unsw_df = load_unsw(unsw_path)

    # 统一标签空间（二分类）
    cicids_df['label'] = cicids_df['label'].map(LABEL_MAPPING_BINARY)
    unsw_df['label'] = unsw_df['label'].map(UNSW_TO_BINARY)

    # 统一特征空间
    cicids_features = align_to_common_schema(cicids_df, feature_config)
    unsw_features = align_to_common_schema(unsw_df, feature_config)

    return cicids_features, unsw_features


def align_to_common_schema(df, config):
    """
    将数据对齐到统一特征空间
    """
    # 特征重命名
    df = df.rename(columns=config.renaming_map)

    # 选择公共特征
    common_features = config.common_features
    df = df[common_features]

    # 归一化
    scaler = config.scaler
    df = scaler.transform(df)

    return df
```

### 6.3 实验脚本

```python
# 实验1: 同数据集
python train.py --train-dataset cicids2017 --test-dataset cicids2017

# 实验2: 跨数据集泛化（主实验）
python train.py --train-dataset cicids2017 --test-dataset unsw_nb15

# 实验3: 反向泛化
python train.py --train-dataset unsw_nb15 --test-dataset cicids2017

# 运行全部实验
python train.py --ablation --cross-dataset
```

---

## 7. 评估指标体系

### 7.1 核心指标

| 指标 | 公式 | 说明 |
|------|------|------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | 总体准确率 |
| **Precision** | TP/(TP+FP) | 预测为正类的准确率 |
| **Recall** | TP/(TP+FN) | 实际正类的检出率 |
| **F1-Score** | 2*Precision*Recall/(Precision+Recall) | 调和平均 |
| **Macro F1** | 各类的F1均值 | 类别均衡指标 |
| **Attack Recall** | 各攻击类的召回率均值 | 核心指标 |
| **Attack Precision** | 各攻击类的精确率均值 | |
| **FAR (False Alarm Rate)** | FP/(FP+TN) | 误报率 |
| **Attack Miss Rate** | 1 - Attack Recall | 漏报率 |

### 7.1.1 不平衡感知阈值指标

以下指标通过对预测分数进行向量化阈值扫描（O(n log n)）计算，专为类别不平衡的 IDS 场景设计：

| 指标 | 说明 |
|------|------|
| **recall_at_far_1pct** | FAR ≤ 1% 约束下的最佳召回率（默认模型选择指标） |
| **recall_at_far_5pct** | FAR ≤ 5% 约束下的最佳召回率 |
| **best_f1** | 所有阈值中的最优 F1（tie-break: 最大 recall → 最小 FAR → 最大阈值） |
| **pr_auc** | Precision-Recall 曲线下面积 |
| **roc_auc** | ROC 曲线下面积 |

这些指标需要传入 `y_score`（模型输出概率）。模型训练时通过 `training.selection_metric` 配置项选择用于保存最佳 checkpoint 的指标，默认为 `recall_at_far_1pct`。

### 7.2 推理延迟指标

| 指标 | 说明 |
|------|------|
| **Latency (ms)** | 单样本推理时间 |
| **Throughput (samples/s)** | 吞吐量 |
| **Batch Latency (ms)** | 批量推理时间 |

```python
def measure_inference_latency(model, dataloader, device, n_batches=10):
    model.eval()
    latencies = []

    with torch.no_grad():
        for i, (features, _) in enumerate(dataloader):
            if i >= n_batches:
                break

            features = features.to(device)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(features)

            if device.type == 'cuda':
                torch.cuda.synchronize()

            latencies.append((time.perf_counter() - start) * 1000)  # ms

    return {
        'mean_latency_ms': np.mean(latencies),
        'p50_latency_ms': np.percentile(latencies, 50),
        'p99_latency_ms': np.percentile(latencies, 99),
    }
```

### 7.3 评估代码

```python
from nids.evaluation.metrics import compute_nids_metrics

# 基础指标（无需概率分数）
metrics = compute_nids_metrics(y_true, y_pred, benign_class=0)

# 完整指标（含阈值扫描指标，需要 y_score）
metrics = compute_nids_metrics(y_true, y_pred, benign_class=0, y_score=y_score)

# 返回 dict 包含:
# accuracy, macro_f1, avg_attack_recall, attack_macro_precision,
# benign_false_alarm_rate, attack_miss_rate, confusion_matrix,
# pr_auc, roc_auc, best_f1, best_f1_threshold,
# recall_at_far_1pct, threshold_at_far_1pct,
# recall_at_far_5pct, threshold_at_far_5pct
```

---

## 8. 项目代码结构设计

### 8.1 文件夹结构

```
flowguard-ids/
├── docs/                          # 文档
│   ├── NIDS_Technical_Design.md   # 本文档
│   └── API_Reference.md
│
├── data/                          # 数据目录
│   ├── raw/                       # 原始数据
│   │   ├── cicids2017/
│   │   └── unsw_nb15/
│   ├── processed/                 # 处理后数据（.npz、scaler、metadata）
│   └── shap_samples/              # 缓存的 SHAP 样本
│
├── nids/                          # 主代码包
│   ├── __init__.py
│   ├── config.py                  # 配置 dataclass 定义 + YAML 加载器
│   │
│   ├── data/                      # 数据处理模块
│   │   ├── __init__.py
│   │   ├── dataset.py             # NIDSDataset + DataLoader 工厂
│   │   ├── preprocessing.py       # 预处理函数
│   │   ├── augmentation.py        # SMOTE / 重采样
│   │   └── cross_dataset.py       # 跨数据集特征对齐
│   │
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── base.py                # 抽象基类
│   │   ├── cnn_bilstm.py          # 基础 CNN-BiLSTM
│   │   ├── cnn_bilstm_se.py       # 主模型（SE 注意力）
│   │   ├── cnn_bilstm_attention.py# 注意力变体（未纳入训练命令）
│   │   ├── classical.py           # Random Forest、XGBoost 封装
│   │   └── registry.py            # 模型名称 → 类映射
│   │
│   ├── training/                  # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py             # 主训练循环 + 评估
│   │   ├── callbacks.py           # Early Stopping
│   │   └── optimizers.py          # 优化器 & 调度器工厂
│   │
│   ├── evaluation/                # 评估模块
│   │   ├── __init__.py
│   │   ├── evaluator.py           # 高级模型评估
│   │   ├── metrics.py             # NIDS 专用指标（含不平衡感知阈值指标）
│   │   └── latency.py             # 推理延迟测量
│   │
│   ├── features/                  # 特征工程模块
│   │   ├── __init__.py
│   │   ├── shap_analysis.py       # SHAPAnalyzer（GradientExplainer / DeepExplainer）
│   │   ├── feature_selector.py    # Top-K / 累积阈值选择
│   │   └── importance.py          # 特征重要性聚合
│   │
│   └── utils/                     # 工具模块
│       ├── __init__.py
│       ├── logging.py             # 结构化日志
│       ├── reproducibility.py     # 随机种子设置
│       ├── io.py                  # JSON / artifact I/O
│       ├── visualization.py       # 论文级图表生成
│       └── process.py             # 子进程工具
│
├── scripts/                       # CLI 入口脚本
│   ├── preprocess.py              # 单数据集预处理
│   ├── preprocess_cross_dataset.py # 跨数据集预处理
│   ├── train.py                   # 训练一个模型 / 全部模型 / 一键训练
│   ├── train_lightweight.py       # 从 Top-K 特征重训轻量模型
│   ├── run_experiments.py         # 批量实验（3 组设定 × 5 模型）
│   ├── evaluate.py                # 评估已保存模型
│   ├── shap_analysis.py           # SHAP 可解释性流程
│   ├── feature_selection.py       # Top-K / 累积特征选择
│   └── export_model.py            # 导出 TorchScript / ONNX
│
├── tests/                         # 单元测试
│   ├── conftest.py                # Pytest fixtures
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_training.py
│   └── test_metrics.py
│
├── configs/                       # 配置文件
│   └── default.yaml               # 唯一配置；跨数据集增强通过 --cross-dataset-enhancements 启用
│
├── Dockerfile                     # CUDA 训练环境
├── requirements.txt
├── setup.py
└── README.md
```

### 8.2 核心文件说明

| 文件 | 职责 |
|------|------|
| `nids/config.py` | 配置类定义 |
| `nids/data/preprocessing.py` | 数据预处理函数 |
| `nids/models/cnn_bilstm_se.py` | 主模型实现 |
| `nids/training/trainer.py` | 训练循环 |
| `nids/features/shap_analysis.py` | SHAP分析 |
| `scripts/preprocess.py` | 预处理入口 |
| `scripts/train.py` | 训练入口 |

---

## 9. 模块接口定义

### 9.1 数据预处理模块

```python
# nids/data/preprocessing.py

def load_dataset(dataset_name: str, data_dir: Path) -> pd.DataFrame:
    """
    加载原始数据集

    Args:
        dataset_name: 'cicids2017' 或 'unsw_nb15'
        data_dir: 数据目录

    Returns:
        DataFrame: 原始数据
    """

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据清洗：去重、处理NaN/Inf、删除无效特征

    Args:
        df: 原始DataFrame

    Returns:
        清洗后的DataFrame
    """

def align_features(cicids_df: pd.DataFrame, unsw_df: pd.DataFrame,
                  config: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    特征对齐

    Args:
        cicids_df: CICIDS2017数据
        unsw_df: UNSW-NB15数据
        config: 特征对齐配置

    Returns:
        对齐后的DataFrame元组
    """

def encode_labels(df: pd.DataFrame, label_col: str,
                 mapping: dict) -> Tuple[np.ndarray, LabelEncoder]:
    """
    标签编码

    Args:
        df: 数据
        label_col: 标签列名
        mapping: 标签映射字典

    Returns:
        编码后的标签数组, 编码器
    """

def split_data(X: np.ndarray, y: np.ndarray,
              train_ratio: float, val_ratio: float,
              stratify: bool = True) -> Tuple[np.ndarray, ...]:
    """
    数据划分

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """

def fit_scaler(X_train: np.ndarray, scaler_type: str = 'minmax') -> Scaler:
    """
    训练归一化器（仅在训练集fit）

    Returns:
        fitted scaler
    """

def transform_features(X: np.ndarray, scaler: Scaler) -> np.ndarray:
    """
    应用归一化

    Returns:
        归一化后的数据
    """
```

### 9.2 模型模块

```python
# nids/models/cnn_bilstm_se.py

class CNNBiLSTMSE(nn.Module):
    def __init__(
        self,
        input_dim: int,           # 特征维度
        num_classes: int,         # 类别数
        conv_channels: List[int], # [64, 128]
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = False,
        use_se: bool = True
    ):
        """
        CNN-BiLSTM-SE模型
        """

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            inputs: (batch_size, feature_dim)

        Returns:
            logits: (batch_size, num_classes) 或 (batch_size,)
        """

    def get_metadata(self) -> dict:
        """返回模型配置元数据"""
```

### 9.3 SHAP特征分析模块

```python
# nids/features/shap_analysis.py

class SHAPAnalyzer:
    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device

    def sample_data(self, X_train: np.ndarray, y_train: np.ndarray,
                   n_samples: int = 2000) -> np.ndarray:
        """
        分层采样用于SHAP分析
        """

    def compute_shap_values(self, X_samples: np.ndarray,
                           background_size: int = 100) -> np.ndarray:
        """
        计算SHAP值
        """

    def compute_feature_importance(self, shap_values: np.ndarray) -> np.ndarray:
        """
        计算特征重要性
        """

    def select_top_k(self, importance: np.ndarray, feature_names: List[str],
                    k: int = 30) -> List[str]:
        """
        选择Top-K特征
        """
```

### 9.4 训练模块

```python
# nids/training/trainer.py

class Trainer:
    def __init__(self, config: TrainingConfig, output_dir: Path):
        self.config = config
        self.output_dir = output_dir

    def fit(self, model: nn.Module,
           train_loader: DataLoader,
           val_loader: DataLoader) -> TrainingSummary:
        """
        训练模型

        Returns:
            TrainingSummary: 包含训练历史、最佳模型路径等
        """

    def evaluate(self, model: nn.Module,
                data_loader: DataLoader) -> EvaluationResult:
        """
        评估模型

        Returns:
            EvaluationResult: 包含损失、指标、预测结果
        """

    def measure_latency(self, model: nn.Module,
                       data_loader: DataLoader) -> dict:
        """
        测量推理延迟
        """
```

### 9.5 数据集类

```python
# nids/data/dataset.py

class NIDSDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test,
                      batch_size: int = 128,
                      num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建DataLoader
    """
```

---

## 10. 开发优先级与实施步骤

### 10.1 开发阶段划分

**阶段1：基础设施搭建（1-2周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| 完善项目文件夹结构 | P0 | 0.5天 |
| 迁移现有代码到模块 | P0 | 2天 |
| 配置管理系统 | P0 | 1天 |
| 跨数据集预处理模块 | P0 | 3天 |
| 数据加载测试 | P0 | 1天 |

**阶段2：模型开发（2周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| 实现CNN-BiLSTM基线 | P0 | 2天 |
| 添加SE模块 | P0 | 1天 |
| 添加Attention模块 | P1 | 1天 |
| 模型注册机制 | P1 | 1天 |
| 单元测试 | P1 | 2天 |

**阶段3：SHAP集成（1周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| SHAP分析模块 | P0 | 2天 |
| 特征选择Pipeline | P0 | 2天 |
| Top-K筛选功能 | P1 | 1天 |
| 结果保存/可视化 | P1 | 1天 |

**阶段4：训练流程（1周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| 完善训练器 | P0 | 2天 |
| Early Stopping | P0 | 1天 |
| Learning Rate Scheduler | P0 | 1天 |
| Checkpoint管理 | P1 | 1天 |
| 训练脚本CLI | P1 | 1天 |

**阶段5：实验与评估（2周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| 同数据集实验 | P0 | 2天 |
| Cross-dataset实验 | P0 | 3天 |
| 指标计算与报告 | P0 | 2天 |
| 可视化 | P1 | 2天 |
| 性能优化 | P1 | 3天 |

**阶段6：传统ML对比（1周）**
| 任务 | 优先级 | 预计工作量 |
|------|--------|-----------|
| Random Forest实现 | P1 | 2天 |
| XGBoost实现 | P1 | 2天 |
| 对比实验 | P1 | 2天 |

### 10.2 里程碑检查点

```
里程碑1 (M1): 第2周周末
  ✓ 完成基础设施搭建
  ✓ 数据可以正常加载

里程碑2 (M2): 第4周周末
  ✓ 完成所有模型实现
  ✓ 基础训练流程可用

里程碑3 (M3): 第5周周末
  ✓ SHAP功能完成
  ✓ 可进行特征选择

里程碑4 (M4): 第6周周末
  ✓ 完整训练流程
  ✓ 生成训练报告

里程碑5 (M5): 第8周周末
  ✓ 完成所有实验
  ✓ 性能达标
```

---

## 11. RTX3060优化策略

### 11.1 硬件适配参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **GPU显存** | 6GB | RTX3060 |
| **Batch Size** | 128 | 可根据显存调整 |
| **Mixed Precision** | True | AMP加速 |
| **DataLoader workers** | 0-2 | 避免IO瓶颈 |

### 11.2 显存优化

```python
# 1. 启用AMP混合精度训练
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda'):
    outputs = model(inputs)
    loss = criterion(outputs, labels)

# 2. 梯度累积（显存受限时使用）
accumulation_steps = 4
effective_batch_size = batch_size * accumulation_steps

# 3. 减少batch内显存占用
def train_with_accumulation(model, dataloader, accumulation_steps):
    model.train()
    optimizer.zero_grad()

    for i, (features, labels) in enumerate(dataloader):
        features = features.to('cuda', non_blocking=True)
        labels = labels.to('cuda', non_blocking=True)

        with torch.amp.autocast('cuda'):
            outputs = model(features)
            loss = criterion(outputs, labels) / accumulation_steps

        loss.backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
```

### 11.3 推理优化

```python
# 1. TorchScript导出
model.eval()
scripted_model = torch.jit.trace(model, example_inputs)
scripted_model.save('model.pt')

# 2. ONNX导出
torch.onnx.export(model, example_inputs, 'model.onnx',
                 input_names=['input'],
                 output_names=['output'],
                 dynamic_axes={'input': {0: 'batch'}})

# 3. 使用TensorRT（可选，需额外安装）
```

### 11.4 性能监控

```python
def monitor_gpu_memory():
    """监控GPU显存使用"""
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                           '--format=csv,noheader,nounits'],
                          capture_output=True, text=True)
    used, total = map(int, result.stdout.strip().split(','))
    print(f"GPU显存: {used}MB / {total}MB ({used/total*100:.1f}%)")
```

---

## 12. 潜在风险与解决方案

### 12.1 技术风险

| 风险 | 影响 | 概率 | 解决方案 |
|------|------|------|----------|
| 特征对齐失败 | 跨数据集实验无法进行 | 高 | 手动建立特征映射表 |
| SHAP计算超时 | 特征选择延迟 | 中 | 减少采样数量，使用近似方法 |
| 显存不足 | 训练中断 | 中 | 降低batch size，使用梯度累积 |
| 类别不平衡严重 | 模型偏向多数类 | 高 | 使用class weights + SMOTE |
| 过拟合 | 泛化能力差 | 高 | 早停、正则化、数据增强 |

### 12.2 数据风险

| 风险 | 解决方案 |
|------|----------|
| 数据集格式不一致 | 统一到NetFlow格式 |
| 标签不一致 | 建立统一映射表 |
| 特征缺失 | 使用默认值或删除该特征 |
| 极端值 | 截断处理 |

### 12.3 实验风险

| 风险 | 解决方案 |
|------|----------|
| 随机性导致结果不稳定 | 设置固定种子，多次实验取平均 |
| 超参数不收敛 | 使用网格搜索或贝叶斯优化 |
| 实验复现困难 | 完整记录配置和随机种子 |

### 12.4 依赖风险

| 风险 | 解决方案 |
|------|----------|
| PyTorch版本不兼容 | 固定版本号 |
| 库依赖冲突 | 使用conda/pip freeze |
| GPU驱动问题 | 使用CPU fallback |

---

## 附录

### A. 配置示例

```yaml
# configs/default.yaml (唯一配置；跨数据集增强通过 CLI --cross-dataset-enhancements 启用)
data:
  train_dataset: cicids2017
  test_dataset: unsw_nb15
  batch_size: 128
  num_workers: 2

model:
  name: cnn_bilstm_se
  input_dim: 55
  num_classes: 2
  conv_channels: [64, 128]
  conv_kernel_sizes: [3, 3]
  conv_pool_sizes: [2, 2]
  lstm_hidden_size: 128
  lstm_num_layers: 2
  dropout: 0.3
  bidirectional: true
  use_se: true
  use_attention: false

training:
  num_epochs: 30
  learning_rate: 0.001
  weight_decay: 0.0001
  optimizer: adam
  scheduler: plateau
  early_stopping_patience: 5

shap:
  n_samples: 2000
  background_size: 100
  top_k: 30
```

### B. 运行命令

```bash
# 数据预处理
python scripts/preprocess.py --dataset cicids2017
python scripts/preprocess_cross_dataset.py

# 训练（同数据集）
python scripts/train.py --config configs/default.yaml --one-click

# 训练（跨数据集：自动启用 AUC+Platt+LS 增强）
python scripts/train.py --config configs/default.yaml --train-dataset cicids2017 --test-dataset unsw_nb15 --one-click --cross-dataset-enhancements

# 批量实验（run_experiments.py 会在跨数据集方向自动加 --cross-dataset-enhancements）
python scripts/run_experiments.py --config configs/default.yaml --profile laptop_3060 --one-click

# SHAP分析
python scripts/shap_analysis.py --model artifacts/best_model.pt

# 评估
python scripts/evaluate.py --model artifacts/best_model.pt \
                           --test-data data/processed/unsw_nb15/test.npz
```

---

**文档结束**

*本技术文档可直接用于指导开发实现。所有代码示例均为Python/PyTorch实现参考。*
