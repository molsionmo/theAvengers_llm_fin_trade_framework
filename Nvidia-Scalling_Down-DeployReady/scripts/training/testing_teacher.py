import os
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
import yaml


# Define extraction functions within this script for simplicity
def extract_answer(text):
    """
    Extracts the answer from the generated text.
    If "Final answer:" is present, extracts the answer following that phrase.
    Otherwise, extracts the first occurrence of an answer option (A-E).
    """
    answer_pattern = r'([A-E])'
    final_answer_pattern = r'answer:\s*([A-E])'

    final_answer_match = re.search(final_answer_pattern, text, re.IGNORECASE)
    if final_answer_match:
        return final_answer_match.group(1).upper()
    else:
        answer_match = re.search(answer_pattern, text, re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()
    return ""

def extract_rationale(text):
    # Case 1: Extract rationale before "The final answer:"
    final_answer_pos = text.lower().find(". The final")
    answer_pos = text.lower().find("answer:")
    equals_pos = text.find("=")
    if answer_pos != -1:
        rationale = text[equals_pos:final_answer_pos].strip() if final_answer_pos != -1 else text[equals_pos:final_answer_pos].strip()
        return rationale

    # Case 2: Extract rationale after '='
    if equals_pos != -1:
        rationale = text[equals_pos + 1:].strip()
        return rationale

    # Case 3: No clear rationale found
    return ""


def load_preprocessed_data(file_path):
    """
    Load preprocessed data from a pickle file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data file not found at {file_path}")

    with open(file_path, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


def main():
    # ---------------------------
    # Configuration and Parameters
    # ---------------------------
    # Path to the preprocessed validation data
    preprocessed_val_file = 'val_flan_aqua.pkl'  # Adjust the path if necessary

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ---------------------------
    # Load Preprocessed Data
    # ---------------------------
    print("Loading preprocessed validation data...")
    val_dataset = load_preprocessed_data(preprocessed_val_file)
    print(f"Number of validation examples: {len(val_dataset)}")

    # ---------------------------
    # Initialize Teacher Model and Tokenizer
    # ---------------------------
    print("Initializing teacher model and tokenizer...")
    teacher_model_name = "google/flan-t5-base"
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher_model = AutoModelForSeq2SeqLM.from_pretrained(teacher_model_name)
    teacher_model.to(device)
    teacher_model.eval()

    # ---------------------------
    # Initialize Variables for Evaluation
    # ---------------------------
    teacher_predictions = []
    teacher_rationales = []
    actual_answers = val_dataset['correct']  # Use 'correct' column

    # ---------------------------
    # Generate Teacher Outputs
    # ---------------------------
    print("Generating teacher outputs for validation data...")
    for example in tqdm(val_dataset, desc="Generating teacher outputs"):
        input_ids = torch.tensor(example['input_ids']).unsqueeze(0).to(device)  # Shape: [1, seq_length]
        attention_mask = torch.tensor(example['attention_mask']).unsqueeze(0).to(device)  # Shape: [1, seq_length]

        with torch.no_grad():
            # Generate output with teacher model
            output_ids = teacher_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=512,  # Adjust as needed
                num_beams=3,
                early_stopping=True
            )

        # Decode the generated output
        output_text = teacher_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        teacher_predictions.append(output_text)

        # Extract rationale and answer
        rationale = extract_rationale(output_text)
        answer = extract_answer(output_text)
        teacher_rationales.append(rationale)

    # ---------------------------
    # Compute Teacher Accuracy
    # ---------------------------
    print("Computing teacher model accuracy...")
    correct_count = 0
    total = len(actual_answers)
    for idx in range(total):
        actual = str(actual_answers[idx]).strip().upper()
        predicted = extract_answer(teacher_predictions[idx])
        if actual == predicted:
            correct_count += 1

    teacher_accuracy = correct_count / total if total > 0 else 0.0
    print(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}% ({correct_count}/{total})")

    # ---------------------------
    # Print Detailed Examples (Optional)
    # ---------------------------
    num_examples_to_show = min(5, total)
    print(f"\nFirst {num_examples_to_show} Validation Examples:\n")
    for idx in range(num_examples_to_show):
        example = val_dataset[idx]
        input_text = teacher_tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        actual_answer = example['correct']
        teacher_output = teacher_predictions[idx]
        teacher_rationale = teacher_rationales[idx]
        teacher_extracted_answer = extract_answer(teacher_output)

        print(f"Example {idx + 1}:")
        print(f"Input: {input_text}")
        print(f"Actual Answer: {actual_answer}")
        print(f"Teacher's Output: {teacher_output}")
        print(f"Teacher's Extracted Answer: {teacher_extracted_answer}")
        print(f"Teacher's Rationale: {teacher_rationale}\n")

    # ---------------------------
    # Save Detailed Results (Optional)
    # ---------------------------
    # You can save the detailed results to a file for further analysis
    # Uncomment the following lines if desired

    report_file = 'teacher_evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write(f"Teacher Model Accuracy: {teacher_accuracy * 100:.2f}% ({correct_count}/{total})\n\n")
        for idx in range(total):
            example = val_dataset[idx]
            input_text = teacher_tokenizer.decode(example['input_ids'], skip_special_tokens=True)
            actual_answer = example['correct']
            teacher_output = teacher_predictions[idx]
            teacher_rationale = teacher_rationales[idx]
            teacher_extracted_answer = extract_answer(teacher_output)
            f.write(f"Example {idx + 1}:\n")
            f.write(f"Input: {input_text}\n")
            f.write(f"Actual Answer: {actual_answer}\n")
            f.write(f"Teacher's Output: {teacher_output}\n")
            f.write(f"Teacher's Extracted Answer: {teacher_extracted_answer}\n")
            f.write(f"Teacher's Rationale: {teacher_rationale}\n\n")
    print(f"Detailed teacher evaluation report saved to {report_file}")


if __name__ == "__main__":
    main()