# actions/logger_util.py

import json
import os
import csv
from datetime import datetime

def log_summary_query(query, section_title, summary, score=None, debug=False):
    CSV_LOG_PATH = os.path.join(os.path.dirname(__file__), "nlu_summary_queries.csv")

    if isinstance(summary, (tuple, list)):
        summary = " ".join(map(str, summary))
    elif not isinstance(summary, str):
        summary = str(summary)

    summary = summary.replace("\n", " ").strip()
    if len(summary) > 500:
        summary = summary[:497] + "..."

    file_exists = os.path.exists(CSV_LOG_PATH)
    with open(CSV_LOG_PATH, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["timestamp", "user_query", "section_title", "score", "summary_response"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            query,
            section_title,
            f"{score:.4f}" if score is not None else "",
            summary
        ])

    if debug:
        print("=" * 40)
        print(f"[SUMMARY LOG] Query: {query}")
        print(f"[SUMMARY LOG] Section: {section_title}")
        if score is not None:
            print(f"[SUMMARY LOG] Score: {score:.4f}")
        print(f"[SUMMARY LOG] Summary:\n{summary}")
        print("=" * 40)

def log_debug(title_path, buffer_text, summary):        
        DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "debug_summary_log.txt")
        
        log_block = []
        log_block.append(f"\n[EXTRACTED SECTION]")
        log_block.append(f"[{datetime.now().isoformat(timespec='seconds')}]")
        log_block.append(f"Title Path: {' > '.join(title_path)}")
        log_block.append(f"Content:\n{buffer_text[:500]}...")        
        log_block.append(f"Summary: {summary[:300]}...\n")
        

        log_text = "\n".join(log_block)

        # Print to console
        print(log_text)

        # Append to file
        with open(DEBUG_LOG_PATH, mode="a", encoding="utf-8") as f:
            f.write(log_text)
    

def log_debug_all(title_path, buffer_text, summary):
    DEBUG_LOG_PATH = os.path.join(os.path.dirname(__file__), "debug_summary_log.json")

    log_entry = {
        #"timestamp": datetime.now().isoformat(timespec='seconds'),
        "title": title_path,
        "summary": summary[:300],
        "content": buffer_text[:500]
    }

    try:
        with open(DEBUG_LOG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(log_entry)
    print(datetime.now().isoformat(timespec='seconds'))
    print(json.dumps(log_entry, indent=2))  # Optional console print

    with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)        

            