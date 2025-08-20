import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import joblib
import numpy as np

# Load the saved model and tokenizer
model_path = 'X:/autism nlp model/trained_models'
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Load the label encoder
label_encoder = joblib.load('X:/autism nlp model/label_encoder.pkl')

# Move the model to the GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# Load the questions from the updated CSV file
questions_df = pd.read_csv('X:/autism nlp model/autism_questions.csv')  # Update the path as necessary

# Get all unique questions
questions = questions_df['Question'].tolist()

# Store probabilities for each answer
probabilities = []

# Function to predict autism type and return probabilities
def predict_autism_type(answer):
    model.eval()
    inputs = tokenizer(answer, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()  # Return probabilities as a numpy array

# Main testing loop
for question in questions:
    print(question)
    answer = input("Your answer: ")
    prob = predict_autism_type(answer)  # Get the probabilities for the current answer
    probabilities.append(prob)  # Store the probabilities

# Convert list of probabilities to a numpy array
probabilities = np.array(probabilities)

# Print the shape and contents of probabilities for debugging
print(f"Probabilities shape: {probabilities.shape}")
print(f"Probabilities: {probabilities}")

# Check if there are any probabilities collected
if probabilities.size == 0:
    print("No probabilities collected. Please check the model predictions.")
else:
    # Calculate the average probabilities for each class
    # Squeeze the probabilities array to remove single-dimensional entries
    probabilities = probabilities.squeeze()  # Adjust the shape if needed
    average_probabilities = probabilities.mean(axis=0)

    # Make sure we don't access an out-of-bounds index
    if average_probabilities.size > 0:
        # Get the predicted label index with the highest probability
        predicted_label_idx = np.argmax(average_probabilities)
        predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]

        # Output the final prediction
        print(f"The model predicts: '{predicted_label}' with a confidence probability of {average_probabilities[predicted_label_idx]:.4f} based on all your answers.")
        
        # Save the final prediction to a file
        with open('X:/autism nlp model/prediction.txt', 'w') as f:
            f.write(f"The model predicts: '{predicted_label}' with a confidence probability of {average_probabilities[predicted_label_idx]:.4f} based on all your answers.\n")
    else:
        print("No average probabilities computed.")

print("Testing complete.")
