import pandas as pd
import json
from openai import OpenAI

client = OpenAI()
file_path = "echr_train.json"

# Read the JSON file
with open(file_path, "r") as file:
    data = json.load(file)

# Extract 'text' values
text_values = [item["text"] for item in data if "text" in item]

# Create a DataFrame
tab_df = pd.DataFrame(text_values, columns=["sentence"]).head(20)
questions_df = pd.DataFrame(columns=["Questions"])

for text in tab_df["sentence"]:
    prompt = f"""Please generate 5 generic questions based on this court case summary:

    {text}
    
    Focus on creating a mix of questions that cover the key facts, legal issues, and implications of the case outcomes.
    """
    completion = client.chat.completions.create(
        model="gpt-4-1106-preview",
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": """
                    You are an experienced lawyer, and you will assist me in generating meaningful questions from court cases in the following JSON format:
                    {"questions": [{"question": "question 1"}, {"question": "question 2"}, {"question": "question 3"}, {"question": "question 4"}, {"question": "question 5"}]}
                """,
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    json_response = completion.choices[0].message.content
    print(json_response)
    print("-" * 100)

    try:
        # Load the JSON response
        data = json.loads(json_response)

        # Extract questions and create a list
        questions_list = [item["question"] for item in data["questions"]]

        # Create a DataFrame with the new questions
        new_df = pd.DataFrame(questions_list, columns=["Questions"])

        # Append to the global DataFrame using pd.concat
        questions_df = pd.concat([questions_df, new_df], ignore_index=True)

        # Save to CSV
        questions_df.to_csv("questions.csv", index=False)
    except Exception as e:
        print(f"An error occurred: {e}")
