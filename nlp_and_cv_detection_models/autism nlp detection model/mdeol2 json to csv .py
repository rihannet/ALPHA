import pandas as pd
import json

# Load JSON data from a file
with open('data.json') as json_file:
    data = json.load(json_file)

# Initialize a list to store the flattened data
data_list = []

# Loop through each autism type in the JSON data
for autism_type, content in data["autism_types"].items():
    # For each question in the current autism type
    for question_id, question_content in content["questions"].items():
        question_text = question_content["question"]
        answers = question_content["answers"]  # Assuming answers are stored under "answers" key

        # Create a set to avoid duplicate answers
        unique_answers = set(answers)

        # Add each unique answer as a separate row
        for answer in unique_answers:
            data_list.append({
                "Label": autism_type,
                "Question_ID": question_id,
                "Question": question_text,
                "Answer": answer
            })

# Create a DataFrame from the list
df = pd.DataFrame(data_list)

# Save the DataFrame to a CSV file
df.to_csv('autism_data.csv', index=False)
