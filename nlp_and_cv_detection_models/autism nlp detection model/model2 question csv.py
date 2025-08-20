import pandas as pd

# Creating a DataFrame with the questions for all autism types
questions_data = {
    "Question_ID": [f"Q{i}" for i in range(1, 8) for _ in range(6)],
    "Question": [
        "How does the child typically respond in social situations?",
        "Describe the child’s communication skills: verbal and non-verbal.",
        "How does the child handle changes in routine or environment?",
        "What specific behaviors, mannerisms, or repetitive actions have you observed?",
        "How would you describe the child’s motor skills or physical coordination?",
        "How does the child express or manage their emotions?",
        "Does the child show sensitivity to specific sounds, lights, or textures?",
    ] * 6,
}

# Creating the DataFrame
questions_df = pd.DataFrame(questions_data)

# Saving the questions to a CSV file
questions_csv_path = 'X:/autism nlp model/autism_questions.csv'
questions_df.to_csv(questions_csv_path, index=False)

questions_csv_path
