import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

data = pd.read_csv('Depression Student Dataset.csv')
data['Have you ever had suicidal thoughts ?'] = data['Have you ever had suicidal thoughts ?'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Family History of Mental Illness'] = data['Family History of Mental Illness'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Depression'] = data['Depression'].apply(lambda x: 1 if x == 'Yes' else 0)
data['Gender'] = data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
data['Sleep Duration'] = data['Sleep Duration'].apply(lambda x: 0 if x == '7-8 hours' else 1 if x == 'Less than 5 hours' else 2 if x == '5-6 hours' else 3)
data['Dietary Habits'] = data['Dietary Habits'].apply(lambda x: 1 if x == 'Healthy' else 2 if x == 'Unhealthy' else 0)

features = list(set(data.columns) - {'Depression'})
X = data[features]
y = data['Depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)

model = Model(input_size=len(features))
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'model.pth')

model.eval()
y_pred_list = []
y_true_list = []
y_prob_list = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        y_prob = model(X_batch)
        y_pred = (y_prob > 0.5).float()
        y_pred_list.append(y_pred)
        y_true_list.append(y_batch)
        y_prob_list.append(y_prob)

y_pred_all = torch.cat(y_pred_list).cpu().numpy()
y_true_all = torch.cat(y_true_list).cpu().numpy()
y_prob_all = torch.cat(y_prob_list).cpu().numpy()

print("Classification Report:")
print(classification_report(y_true_all, y_pred_all, target_names=["No Depression", "Depression"]))

roc_auc = roc_auc_score(y_true_all, y_prob_all)
print(f"ROC-AUC Score: {roc_auc:.4f}")

cm = confusion_matrix(y_true_all, y_pred_all)
fpr, tpr, _ = roc_curve(y_true_all, y_prob_all)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
