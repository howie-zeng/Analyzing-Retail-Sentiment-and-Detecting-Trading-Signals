import os
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Pool
import multiprocessing


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, LSTM
from keras import utils
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        return text, label
    
class SentimentClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=100):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1])
        x = self.fc(x)
        x = self.sigmoid(x)
        return x.squeeze()
    


def train(train_loader, test_loader, embedding_matrix, batch_size, num_epochs=6, learning_rate=0.001, save_path="model/"):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentClassifier(embedding_matrix).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    for epoch in range(num_epochs):
        model_name = f"rnn_bs{batch_size}_epoch{epoch+1}.pt"
        model_path = os.path.join(save_path, model_name)

        if os.path.exists(model_path):
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path))
            continue

        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct_preds = 0
        total_preds = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = torch.round(outputs)
                correct_preds += (preds == y.squeeze()).sum().item()
                total_preds += y.size(0)
        accuracy = correct_preds / total_preds

        torch.save(model.state_dict(), model_path)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy:.4f}")




def evaluate_model(model, test_loader, y_test, embedding_matrix, original_texts, threshold=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Predict probabilities on test data
    model.eval()
    y_pred_probs = []
    mislabeled_examples = []  # List to store mislabeled examples

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device)
            outputs = model(x)
            y_pred_probs.extend(outputs.cpu().numpy())

    # Convert probabilities to binary predictions
    y_pred = [1 if prob > threshold else 0 for prob in y_pred_probs]

    # Convert y_test to a flat list
    y_true = y_test.squeeze().tolist()

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # F1-Score
    f1 = f1_score(y_true, y_pred)
    print(f"F1-Score: {f1:.4f}")

    # ROC-AUC
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_percent, annot=True, fmt='.2%', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report
    report = classification_report(y_true, y_pred, target_names=['NEGATIVE', 'POSITIVE'])
    print(report)

    # Collect mislabeled examples
    for i, (true_label, pred_label, original_text) in enumerate(zip(y_true, y_pred, original_texts)):
        if pred_label != true_label:
            mislabeled_examples.append({
                "original_text": original_text,
                "predicted": pred_label,
                "true": true_label
            })

    return mislabeled_examples




def categorize_errors(mislabeled_examples):
    short_texts = []
    texts_with_not = []
    other_errors = []

    for example in mislabeled_examples:
        text = example["original_text"]
        if len(text.split()) < 5:  # Assuming a short text has less than 5 words
            short_texts.append(text)
        elif "not" in text:
            texts_with_not.append(text)
        else:
            other_errors.append(text)

    return short_texts, texts_with_not, other_errors

def quantify_errors(short_texts, texts_with_not, other_errors):
    print(f"Number of short texts: {len(short_texts)}")
    print(f"Number of texts with 'not': {len(texts_with_not)}")
    print(f"Number of other errors: {len(other_errors)}")

def deep_dive(short_texts, texts_with_not):
    print("\nExamples of short texts:")
    for text in short_texts[:5]:  # Print first 5 examples
        print(text)

    print("\nExamples of texts with 'not':")
    for text in texts_with_not[:5]:  # Print first 5 examples
        print(text)

def print_mislabeled_examples(mislabeled_examples):
    for i, example in enumerate(mislabeled_examples, start=1):
        print(f"Example {i}:")
        print(f"Original Text: {example['original_text']}")
        print(f"Predicted Label: {example['predicted']}")
        print(f"True Label: {example['true']}")
        print("\n")


def clean_directory(directory_path):
    """
    Clean up files in the specified directory.

    Args:
    directory_path (str): The path to the directory to be cleaned.

    Returns:
    None
    """
    if os.path.exists(directory_path):
        files = os.listdir(directory_path)
        for file in files:
            file_path = os.path.join(directory_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete: {file_path}, Error: {e}")
    else:
        print(f"Directory not found: {directory_path}")
