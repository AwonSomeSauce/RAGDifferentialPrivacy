import json
import os
import sys
import pandas as pd

# Importing custom modules
from llama_index import Document, VectorStoreIndex, ServiceContext
from llama_index.llms import OpenAI
from trulens_eval import Tru
from utils import get_prebuilt_trulens_recorder
from utils import build_sentence_window_index
from utils import get_sentence_window_query_engine
from utils import build_automerging_index
from utils import get_automerging_query_engine

# Adjusting the system path to include the root directory
current_script_path = os.path.abspath(__file__)
root_directory = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, root_directory)

from mechanisms.santext import SanText
from mechanisms.custext import CusText
from mechanisms.distext import DisText
from mechanisms.custext_modified import CusTextModified
from mechanisms.ktext import kText
from mechanisms.detectors.santext_detector import SanTextDetector
from mechanisms.detectors.custext_detector import CusTextDetector
from mechanisms.detectors.presidio_detector import PresidioDetector

# Constants
WORD_EMBEDDING = "glove"
WORD_EMBEDDING_PATH = "glove.840B.300d.txt"
TOP_K = 20
EPSILON = 1.0
P = 0.3
DISTANCE = 5.5
TAB_FILE_PATH = "echr_train.json"
MODELS = ["gpt-3.5-turbo-0125", "gpt-4-1106-preview", "gpt-3.5-turbo-1106"]


def get_results(answer_relevance, context_relevance, groundedness):
    """Calculate and return the average for each specified column"""
    return (answer_relevance.mean(), context_relevance.mean(), groundedness.mean())


def read_json_file(file_path):
    """Read a JSON file and return the data"""
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_text_values(data):
    """Extract 'text' values from data"""
    return [item["text"] for item in data if "text" in item]


def chunks(lst, n):
    """Yield successive n-sized chunks from lst"""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def process_mechanisms(tab_df, mechanisms):
    """Process data through various mechanisms and return the results DataFrame"""
    results_df = pd.DataFrame(
        columns=["Mechanism", "Answer Relevance", "Context Relevance", "Groundedness"]
    )

    for mechanism in mechanisms:
        print(f"PROCESSING: {mechanism}")
        print("-" * 100)

        if mechanism.endswith("Plus"):
            detector = (
                SanTextDetector(0.9) if "SanText" in mechanism else CusTextDetector()
            )
            sanitizer = (
                SanText(WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, P, detector)
                if "SanText" in mechanism
                else (
                    CusText(
                        WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, TOP_K, detector
                    )
                    if "CusText" in mechanism
                    else DisText(
                        WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, DISTANCE, detector
                    )
                )
            )
            tab_df = sanitizer.sanitize(tab_df)
        elif mechanism.endswith("Presidio"):
            detector = PresidioDetector()
            sanitizer = (
                SanText(WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, P, detector)
                if "SanText" in mechanism
                else (
                    CusText(
                        WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, TOP_K, detector
                    )
                    if "CusText" in mechanism
                    else DisText(
                        WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, DISTANCE, detector
                    )
                )
            )
            tab_df = sanitizer.sanitize(tab_df)

    return results_df


def evaluate_mechanism(tab_df, mechanism, results_df):
    """Evaluate the mechanism and append results to the DataFrame"""
    document = Document(text="\n\n".join([doc for doc in tab_df["sanitized sentence"]]))
    llm = OpenAI(
        organization="org-5FDaIDe2hzj7FGqPPiL6V4Jk",
        model=MODELS[2],
    )
    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model="local:BAAI/bge-small-en-v1.5"
    )
    index = VectorStoreIndex.from_documents([document], service_context=service_context)
    query_engine = index.as_query_engine()
    tru = Tru()
    tru.reset_database()

    tru_recorder = get_prebuilt_trulens_recorder(query_engine, app_id=mechanism)
    eval_questions = read_lines("trulens_evaluation/tab_eval_questions.txt")

    with tru_recorder as recording:
        for batch in chunks(eval_questions, 5):
            responses = [query_engine.query(question) for question in batch]

    records, feedback = tru.get_records_and_feedback(app_ids=[mechanism])
    records.to_csv(f"results/{mechanism}.csv", index=False)
    results_row = {
        "Mechanism": mechanism,
        "Answer Relevance": records["Answer Relevance"].mean(),
        "Context Relevance": records["Context Relevance"].mean(),
        "Groundedness": records["Groundedness"].mean(),
    }
    return pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)


def auto_merging_retrieval(tab_df, mechanism, results_df):
    documents = [
        Document(text=doc, id_=idx)
        for idx, doc in enumerate(tab_df["sanitized sentence"])
    ]
    llm = OpenAI(model="gpt-3.5-turbo-1106", temperature=0.1)
    automerging_index = build_automerging_index(
        documents,
        llm,
        embed_model="local:BAAI/bge-small-en-v1.5",
        save_dir="merging_index",
    )
    automerging_query_engine = get_automerging_query_engine(
        automerging_index,
    )
    tru = Tru()
    tru.reset_database()
    application = f"{mechanism} with Automerging Query Engine"
    tru_recorder_automerging = get_prebuilt_trulens_recorder(
        automerging_query_engine, app_id=application
    )
    eval_questions = read_lines("trulens_evaluation/tab_eval_questions.txt")
    with tru_recorder_automerging as recording:
        for batch in chunks(eval_questions, 5):
            responses = [automerging_query_engine.query(question) for question in batch]

    records, feedback = tru.get_records_and_feedback(app_ids=[application])
    records.to_csv(f"results/auto_merging_retrieval/{mechanism}.csv", index=False)
    results_row = {
        "Mechanism": mechanism,
        "Answer Relevance": records["Answer Relevance"].mean(),
        "Context Relevance": records["Context Relevance"].mean(),
        "Groundedness": records["Groundedness"].mean(),
    }
    return pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)


def read_lines(file_path):
    """Read lines from a file and return them as a list"""
    with open(file_path, "r", encoding="utf-8") as file:
        return [line.strip() for line in file]


# Main script execution
if __name__ == "__main__":
    data = read_json_file(TAB_FILE_PATH)
    text_values = extract_text_values(data)
    tab_df = pd.DataFrame(text_values, columns=["sentence"])

    detector = PresidioDetector()
    mechanism = kText(WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, P, detector, 10)
    results_df = mechanism.sanitize(tab_df)
    results_df.to_csv("sanitized_tab/kText Presidio.csv", index=False)

    # mechanisms = [
    #     "SanText Plus",
    #     "CusText Plus",
    #     "DisText Plus",
    #     "SanText Presidio",
    #     "CusText Presidio",
    #     "DisText Presidio",
    # ]

    # results_df = process_mechanisms(tab_df, mechanisms)
    # for mechanism in mechanisms:
    #     sanitized_tab_df = pd.read_csv(f"sanitized_datasets/{mechanism}.csv")
    #     results_df = pd.DataFrame(
    #         columns=[
    #             "Mechanism",
    #             "Answer Relevance",
    #             "Context Relevance",
    #             "Groundedness",
    #         ]
    #     )
    #     results_df = auto_merging_retrieval(sanitized_tab_df, mechanism, results_df)
    #     print(results_df)

    # mechanism = "kText Presidio"
    # sanitized_tab_df = pd.read_csv(f"sanitized_datasets/{mechanism}.csv")
    # results_df = pd.DataFrame(
    #     columns=[
    #         "Mechanism",
    #         "Answer Relevance",
    #         "Context Relevance",
    #         "Groundedness",
    #     ]
    # )
    # results_df = evaluate_mechanism(sanitized_tab_df, mechanism, results_df)
    # print(results_df)

    # detector = SanTextDetector(0.9)
    # mechanism = SanText(WORD_EMBEDDING, WORD_EMBEDDING_PATH, EPSILON, P, detector)
    # results_df = mechanism.sanitize(tab_df)
    # results_df.to_csv("random_santext.csv", index=False)
