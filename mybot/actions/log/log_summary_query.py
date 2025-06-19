import csv
from datetime import datetime
import os



def log_summary_query(query, section_title, summary):
    CSV_LOG_PATH =   os.path.join(os.path.dirname(__file__), "nlu_summary_queries.csv" ) 
    if isinstance(summary, tuple):
        summary = summary[1]  # or str(summary)

    # Create file with header if not exists
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "user_query", "section_title", "summary_response"])

    # Append new row
    with open(CSV_LOG_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            section_title,
            summary.replace("\n", " ")  # Clean newlines, # ❌ this fails if summary is a tuple
                
        ])      