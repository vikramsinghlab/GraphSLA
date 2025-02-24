import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('final_input_lncrna_SM.csv')

# Verify the column names
print("Columns in CSV file:", df.columns)


drug_map = {drug: idx for idx, drug in enumerate(df['drug'].unique())}
protein_map = {protein: idx for idx, protein in enumerate(df['mirna'].unique(), len(drug_map))}

# Create edge list, labels, and features
edges = []
labels = []
drug_features = []
mirna_features = []

for _, row in df.iterrows():
    drug_idx = drug_map[row['drug']]
    protein_idx = protein_map[row['mirna']]
    edges.append([drug_idx, protein_idx])
    labels.append(row['interaction'])
    drug_features.append([float(x) for x in row['drug_features'].split(',')])
    mirna_features.append([float(x) for x in row['mirna_features'].split(',')])


edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
labels = torch.tensor(labels, dtype=torch.float)
drug_features = torch.tensor(drug_features, dtype=torch.float)
mirna_features = torch.tensor(mirna_features, dtype=torch.float)


num_drug_features = drug_features.shape[1]
num_mirna_features = mirna_features.shape[1]


x = torch.cat([drug_features, mirna_features], dim=1)


data = Data(x=x, edge_index=edge_index)

# Define the GAT model with more layers
class GAT(nn.Module):
    def __init__(self, input_dim, num_heads=4):
        super(GAT, self).__init__()
        self.conv1 = GATConv(input_dim, 128, heads=num_heads, concat=True)
        self.conv2 = GATConv(128 * num_heads, 128, heads=num_heads, concat=True)
        self.conv3 = GATConv(128 * num_heads, 64, heads=num_heads, concat=True)
        self.conv4 = GATConv(64 * num_heads, 64, heads=1, concat=False)  
        self.fc = nn.Linear(64, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        return x

    def predict(self, x, edge_index):
        x = self.forward(x, edge_index)
        return self.fc(x).squeeze()


model = GAT(x.shape[1], num_heads=4)  
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()
    optimizer.zero_grad()
    out = model.predict(data.x, data.edge_index)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


for epoch in range(10):  
    loss = train()
    if (epoch+1) % 1 == 0:
        print(f'Epoch {epoch+1}, Loss: {loss:.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model.predict(data.x, data.edge_index)).numpy()
    
    
# Calculate AUC-ROC
roc_auc = roc_auc_score(labels.numpy(), predictions)
print(f'AUC-ROC: {roc_auc:.3f}')

# Calculate accuracy, confusion matrix, sensitivity, specificity, and F1 score
predicted_labels = (predictions > 0.5).astype(int)
accuracy = accuracy_score(labels.numpy(), predicted_labels)

# Confusion matrix
conf_matrix = confusion_matrix(labels.numpy(), predicted_labels)
tn, fp, fn, tp = conf_matrix.ravel()

# Sensitivity and specificity
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# F1 score
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = sensitivity 
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

# Print metrics
print(f'Accuracy: {accuracy:.3f}')
print(f'Sensitivity: {sensitivity:.3f}')
print(f'Specificity: {specificity:.3f}')
print(f'F1 Score: {f1_score:.3f}')
print('Confusion Matrix:')
print(conf_matrix)

# Calculate AUC-ROC
roc_auc = roc_auc_score(labels.numpy(), predictions)
print(f'AUC-ROC: {roc_auc:.3f}')

# Calculate accuracy and confusion matrix
predicted_labels = (predictions > 0.5).astype(int)
accuracy = accuracy_score(labels.numpy(), predicted_labels)
print(f'Accuracy: {accuracy:.3f}')

conf_matrix = confusion_matrix(labels.numpy(), predicted_labels)
print('Confusion Matrix:')
print(conf_matrix)

# Plot AUC-ROC curve
fpr, tpr, _ = roc_curve(labels.numpy(), predictions)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Save the model
torch.save(model.state_dict(), 'gat_SM_lncRNA_repurposing.pth')
