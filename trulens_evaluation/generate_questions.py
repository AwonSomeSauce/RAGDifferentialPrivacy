import pandas as pd
import json
from openai import OpenAI


def generate_questions(text, client):
    """Generate questions based on the given text using OpenAI"""
    prompt = f"""Please generate 5 generic questions based on this court case summary:

    {text}

    Focus on creating a mix of questions that cover the key facts, legal issues, and implications of the case outcomes.
    """
    try:
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
        data = json.loads(json_response)
        return [item["question"] for item in data["questions"]]
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def main():
    client = OpenAI()
    tab_file_path = "echr_train.json"

    # Read the JSON file
    with open(tab_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract 'text' values
    text_values = [item["text"] for item in data if "text" in item]

    # Create a DataFrame
    tab_df = pd.DataFrame(text_values, columns=["sentence"]).head(30)
    questions_dfs = []

    for text in tab_df["sentence"]:
        questions = generate_questions(text, client)
        questions_df = pd.DataFrame(questions, columns=["Questions"])
        questions_dfs.append(questions_df)

    # Combine all question DataFrames
    all_questions_df = pd.concat(questions_dfs, ignore_index=True)

    # Save to CSV
    all_questions_df.to_csv("questions.csv", index=False)


if __name__ == "__main__":
    main()
