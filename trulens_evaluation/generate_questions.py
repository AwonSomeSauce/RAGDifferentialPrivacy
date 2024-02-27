import json
import time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI


def generate_questions(text, client):
    """Generate questions based on the given text using OpenAI"""
    prompt = f"""Please generate 5 generic questions based on this court case summary:

    {text}

    Focus on creating a mix of questions that cover the key facts, legal issues, and implications of the case outcomes.
    """
    try:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
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
    client = OpenAI(organization="org-5FDaIDe2hzj7FGqPPiL6V4Jk")
    tab_file_path = "echr_train.json"
    questions_file_path = "trulens_evaluation/tab_eval_questions.txt"

    # Read the JSON file
    with open(tab_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Extract 'text' values
    text_values = [item["text"] for item in data if "text" in item]

    # Create a DataFrame
    tab_df = pd.DataFrame(text_values, columns=["sentence"]).iloc[34:]

    for text in tqdm(tab_df["sentence"]):  # Wrap the iterable in tqdm
        questions = generate_questions(text, client)

        with open(questions_file_path, "a") as f:
            for question in questions:
                f.write(question + "\n")

        time.sleep(5)


if __name__ == "__main__":
    main()
