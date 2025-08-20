import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib  # For saving the label encoder

# Load the dataset
df = pd.read_csv('autism_data.csv')

# Check the data
print(df.head())

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

# Encode the labels
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])

# Save the label encoder to a file
joblib.dump(label_encoder, 'label_encoder.pkl')

# Split the dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenization function
def tokenize_data(df):
    return tokenizer(
        df['Answer'].tolist(),
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=128
    )

# Tokenize training and testing data
train_encodings = tokenize_data(train_df)
test_encodings = tokenize_data(test_df)

# Create a PyTorch Dataset
class AutismDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Create datasets
train_dataset = AutismDataset(train_encodings, train_df['Label'].values)
test_dataset = AutismDataset(test_encodings, test_df['Label'].values)

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=len(label_encoder.classes_))

# Move the model to the GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Training parameters
batch_size = 56
num_epochs = 1000
learning_rate = 5e-5

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Lists to store training and validation metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_correct = 0

    # Training
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_correct += (predictions == labels).sum().item()

    avg_train_loss = total_loss / len(train_loader)
    train_accuracy = total_correct / len(train_df)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=-1)
            val_correct += (predictions == labels).sum().item()

    avg_val_loss = val_loss / len(test_loader)
    val_accuracy = val_correct / len(test_df)

    # Append metrics
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)

    print(f"Epoch {epoch + 1}/{num_epochs} - "
          f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
          f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# Save the model and tokenizer
model.save_pretrained('trained_models')
tokenizer.save_pretrained('trained_models')

print("Training complete and model saved.")
