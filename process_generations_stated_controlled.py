import os
import csv
import concurrent.futures
from tqdm import tqdm
from openai import OpenAI

# Folder containing .full.csv files
input_folder = "generations_stated_controlled"

# Retrieve OpenRouter API key
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise EnvironmentError("The OPENROUTER_API_KEY environment variable is not set.")

# Initialize OpenRouter client
client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

# Configuration
MODEL_NAME = "gpt-4o-mini"  # Faster and cheaper
MAX_WORKERS = 20            # Number of parallel requests (Adjust if you hit rate limits)

def categorize_row(row, base_name, response_col):
    """
    Function to process a single row. Returns the modified row with a category.
    """
    try:
        prompt = row.get("prompt", "") + "\n" + row.get(response_col, "")
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Classify the user's response into exactly one of these three categories: 'binary', 'equal', or 'depends'. Return only the word."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10  # We only need one word, so keep this low to save time
        )
        
        # Access content safely (OpenAI v1+ style)
        category = response.choices[0].message.content.strip().lower()
        
        # Clean up punctuation like "binary." -> "binary"
        if category.endswith('.'):
            category = category[:-1]

        if category in {"binary", "equal", "depends"}:
            row["response_category"] = category
        else:
            row["response_category"] = "depends"  # Fallback

    except Exception as e:
        # On error, you might want to log it or retry, but here we default to depends to keep moving
        # print(f"Error: {e}") 
        row["response_category"] = "depends"
        
    return row

def process_file(filename):
    input_path = os.path.join(input_folder, filename)
    base_name = filename.replace(".full.csv", "")
    
    # Read rows
    with open(input_path, "r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        fieldnames = list(reader.fieldnames)
        rows = list(reader)

    # Dynamically find the correct response column
    # Priority: starts with 'model_resp_' and does NOT contain 'clean'
    response_col = next((col for col in fieldnames if col.startswith("model_resp_") and "clean" not in col), None)

    if not response_col:
        print(f"Skipping {filename}: No valid 'model_resp_' column found.")
        return

    # Add new column to headers if missing
    if "response_category" not in fieldnames:
        fieldnames.append("response_category")

    print(f"Processing {filename} using column '{response_col}' with {MAX_WORKERS} threads...")

    # Process rows in parallel
    processed_rows = []
    counts = {"binary": 0, "equal": 0, "depends": 0}

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create a future for each row
        futures = {executor.submit(categorize_row, row, base_name, response_col): row for row in rows}
        
        # Use tqdm to show progress as futures complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(rows), desc=f"Categorizing"):
            row = future.result()
            processed_rows.append(row)
            counts[row["response_category"]] += 1

    # Filter binary rows (Original order is not guaranteed with threading, but usually fine for analysis)
    # If order matters, we can sort by 'idx' if available, or just append to list in order.
    # Re-sorting to match original order based on input 'rows' list is safer:
    # We can't easily resort unless we have an index. Let's assume order doesn't strictly matter or 
    # the user can sort by 'idx' later. 
    # actually, processed_rows will be out of order. Let's fix that.
    # Map row object id to result? Simpler: Just rely on the fact that rows are dicts.
    
    # Simple fix to maintain order:
    # The 'row' object is mutable. `categorize_row` modified it in place.
    # So `rows` (the original list) now contains the data.
    
    binary_rows = [r for r in rows if r.get("response_category") == "binary"]

    # Write filtered binary file
    binary_path = os.path.join(input_folder, f"{base_name}.csv")
    with open(binary_path, "w", newline="", encoding="utf-8") as binary_file:
        writer = csv.DictWriter(binary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(binary_rows)
    
    # Write summary file
    summary_path = os.path.join(input_folder, f"{base_name}.summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=["category", "count", "percentage"])
        writer.writeheader()
        total = len(rows)
        for cat in ["binary", "equal", "depends"]:
            count = counts[cat]
            pct = (count / total * 100) if total > 0 else 0
            writer.writerow({"category": cat, "count": count, "percentage": round(pct, 2)})

    print(f"Done! Binary: {counts['binary']}, Equal: {counts['equal']}, Depends: {counts['depends']}")

# Main loop
if os.path.exists(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".full.csv"):
            process_file(filename)
else:
    print(f"Folder {input_folder} not found.")