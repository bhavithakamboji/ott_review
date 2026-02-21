import pandas as pd

data = pd.read_csv("train.csv")

# Select required columns
multi_label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

final_data = data[["comment_text"] + multi_label_columns]
final_data.columns = ["review"] + multi_label_columns

# Optional: smaller subset
#final_data = final_data.sample(n=5000, random_state=42)

final_data.to_csv("dataset.csv", index=False)

print("Multi-label dataset created successfully!")