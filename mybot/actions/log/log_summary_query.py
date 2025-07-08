import csv
from datetime import datetime
import os



def log_summary_query(query, section_title, summary,related_sources):
    CSV_LOG_PATH =   os.path.join(os.path.dirname(__file__), "nlu_summary_queries.csv" ) 
    # Normalize summary
    if isinstance(summary, (tuple, list)):
        summary = " ".join(str(s) for s in summary)
    else:
        summary = str(summary)

    # Normalize related_sources
    if isinstance(related_sources, (tuple, list)):
        related_str = ", ".join(str(s) for s in related_sources)
    else:
        related_str = str(related_sources)    
    

    # Create file with header if not exists
    if not os.path.exists(CSV_LOG_PATH):
        with open(CSV_LOG_PATH, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "user_query", "section_title", "summary_response","related_topics"])

    # Append new row
    with open(CSV_LOG_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            section_title,
            summary.replace("\n", " "),  # Clean newlines, # ❌ this fails if summary is a tuple
            related_str
        ])      