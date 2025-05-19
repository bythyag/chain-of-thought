import json
import re

def extract_answer_from_original(original_answer_str):
    """
    Extracts the final numerical answer from the original_answer string.
    Example: "#### 72" -> "72"
    """
    match = re.search(r'####\s*([0-9.]+)', original_answer_str)
    if match:
        return match.group(1)
    return None

def extract_answer_from_text(answer_text_str):
    """
    Extracts the final numerical answer from the answer_text string.
    Example: "The answer is 72." -> "72"
    Example: "The answer is $72." -> "72"
    Example: "The answer is 990.00." -> "990.00"
    """
    match = re.search(r'The answer is \$?([\d,]+\.?\d*)', answer_text_str)
    if match:
        # Remove commas and trailing dots
        number = match.group(1).replace(',', '').rstrip('.')
        return number
    return None

def evaluate_answers(file_path):
    """
    Reads a JSON file, compares original_answer and answer_text,
    and prints the evaluation.
    Returns the results list and the number of matches.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = []
    all_match = True
    match_count = 0

    for i, item in enumerate(data):
        question = item.get("question")
        original_answer_str = item.get("original_answer")
        answer_text_str = item.get("generated_answer")

        if not question or not original_answer_str or not answer_text_str:
            print(f"Warning: Missing data in item {i+1}. Skipping.")
            results.append({
                "question_number": i + 1,
                "question": question,
                "status": "Skipped - Missing data"
            })
            all_match = False
            continue

        extracted_original = extract_answer_from_original(original_answer_str)
        extracted_text = extract_answer_from_text(answer_text_str)

        match = False
        if extracted_original is not None and extracted_text is not None:
            # Compare as floats to handle cases like "72" and "72.00"
            try:
                if float(extracted_original) == float(extracted_text):
                    match = True
                    match_count += 1
            except ValueError:
                # Fallback to string comparison if conversion fails (e.g., unexpected format)
                if extracted_original == extracted_text:
                    match = True
                    match_count += 1
        
        if not match:
            all_match = False

        print(f"Question {i+1}:")
        print(f"  Match: {match}, extracted text: {extracted_text}, original text: {extracted_original}\n")

    print("--- Summary ---")
    if all_match:
        print("All extracted answers match!")
    else:
        print(f"{match_count} out of {len(data)} extracted answers match. See details above.")
    
    return results, match_count

if __name__ == "__main__":
    # Path to your JSON file
    json_file_path = '/Users/thyag/Desktop/codes/model_outputs/chat_response_qwen3-4b-w-system-msg.json'
    evaluation_results, num_matches = evaluate_answers(json_file_path)
    print(f"Number of matching answers: {num_matches}")
    # You can further process evaluation_results if needed