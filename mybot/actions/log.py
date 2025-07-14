import csv
import os
from datetime import datetime
from typing import Optional, Union, List, Tuple

# --- Constants ---
LOG_FILE_PATH = os.path.join(os.path.dirname(__file__), "log", "nlu_summary_queries.csv")
MAX_SUMMARY_LENGTH = 500


def log_summary_query(
    query: str,
    section_title: str,
    summary: Union[str, list, tuple],
    score: Optional[float] = None,
    debug: bool = False,
):
    """
    Logs the summary of a user query interaction to a CSV file.

    This function records the user's query, the title of the retrieved section,
    the generated summary, and an optional confidence score. It also provides
    a debug mode to print the log to the console.

    Args:
        query: The user's original query text.
        section_title: The title or identifier of the knowledge base section used.
        summary: The generated summary text. Can be a string, list, or tuple.
        score: An optional confidence score for the retrieval or generation.
        debug: If True, prints the log entry to the console.
    """
    # --- Data Sanitization ---
    if isinstance(summary, (list, tuple)):
        summary_text = " ".join(map(str, summary))
    else:
        summary_text = str(summary)

    summary_text = summary_text.replace("\n", " ").strip()
    if len(summary_text) > MAX_SUMMARY_LENGTH:
        summary_text = summary_text[: MAX_SUMMARY_LENGTH - 3] + "..."

    # --- CSV Logging ---
    file_exists = os.path.exists(LOG_FILE_PATH)
    try:
        with open(LOG_FILE_PATH, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(
                    ["timestamp", "user_query", "section_title", "score", "summary_response"]
                )
            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    query,
                    section_title,
                    f"{score:.4f}" if score is not None else "",
                    summary_text,
                ]
            )
    except IOError as e:
        print(f"Error writing to log file: {e}")

    # --- Debug Console Output ---
    if debug:
        print("=" * 40)
        print(f"[SUMMARY LOG] Query: {query}")
        print(f"[SUMMARY LOG] Section: {section_title}")
        if score is not None:
            print(f"[SUMMARY LOG] Score: {score:.4f}")
        print(f"[SUMMARY LOG] Summary:\n{summary_text}")
        print("=" * 40)
