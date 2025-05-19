import openai
import sqlite3
from typing import List, Tuple, Dict


# Set your OpenAI API key
openai.api_key = ""

# Allowed relations and reference set for semantic accuracy
allowed_relations = {"AND", "OR", "HELP", "HURT"}
reference_set = {
    "Security": [("Confidentiality", "AND", "Integrity"), ("Integrity", "AND", "Availability")],
    "Usability": [("AvoidRestrictiveAccess", "HURT", "Security")],
    "Performance": [("TimePerformance", "OR", "SpacePerformance")]
}

# Function to clean up any introductory or extraneous text in the output
def clean_generated_output(generated_sig: str) -> str:
    lines = generated_sig.splitlines()
    filtered_lines = [line for line in lines if not line.startswith("SIG") and line.strip()]
    return "\n".join(filtered_lines)

# Function to generate SIGs using a single example-based prompt
def generate_sig_with_example(prompt, max_output_tokens=300, n=5):
    generated_sigs = []
    for i in range(n):
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant knowledgeable in SIG modeling."},
                      {"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=0.2
        )
        
        generated_sig = clean_generated_output(response.choices[0].message['content'].strip())
        print(f"Raw Output {i+1}:\n{generated_sig}\n")
        generated_sigs.append(generated_sig)
    
    return generated_sigs

# Preprocessing and Parsing Functions

def preprocess_output(sig: str) -> str:
    lines = sig.splitlines()
    cleaned_lines = [line.lstrip("0123456789. ").replace("-", "").strip() for line in lines if line.strip() and ":" in line]
    return " | ".join(cleaned_lines)

def parse_sig(sig: str) -> bool:
    lines = sig.split(" | ")
    for line in lines:
        parts = line.split(":")
        if len(parts) != 2:
            return False

        main_category = parts[0].strip()
        rest = parts[1].strip()
        
        if not main_category.isalpha():
            return False

        tokens = rest.split()
        if len(tokens) < 3:
            return False

        expecting_relation = False
        for token in tokens:
            if expecting_relation:
                if token not in allowed_relations:
                    return False
                expecting_relation = False
            else:
                if not token.isalpha():
                    return False
                expecting_relation = True

        if tokens[-1] in allowed_relations:
            return False

    return True

def syntactic_accuracy_test(generated_sigs: List[str]) -> float:
    valid_count = 0
    for i, sig in enumerate(generated_sigs, start=1):
        preprocessed_sig = preprocess_output(sig)
        is_valid = parse_sig(preprocessed_sig)
        print(f"Example {i}: {preprocessed_sig} - {'Valid' if is_valid else 'Invalid'}")
        if is_valid:
            valid_count += 1
    
    accuracy = valid_count / len(generated_sigs) if generated_sigs else 0
    print(f"\nTotal Syntactic Accuracy: {accuracy * 100:.2f}%\n")
    return accuracy

def parse_semantics(line: str) -> Tuple[str, List[Tuple[str, str, str]]]:
    if ":" not in line or not line.strip():
        print(f"Skipping invalid line for semantics: {line}")
        return "", []
    
    main_category, rest = line.split(":", 1)
    main_category = main_category.strip()
    
    tokens = rest.split()
    if not tokens:
        print(f"Skipping empty relationship line: {line}")
        return "", []

    relationships = []
    current_subgoal = tokens[0]
    i = 1
    while i < len(tokens) - 1:
        relation = tokens[i]
        next_subgoal = tokens[i + 1]
        if relation in allowed_relations:
            relationships.append((current_subgoal, relation, next_subgoal))
            current_subgoal = next_subgoal
            i += 2
        else:
            i += 1
    
    return main_category, relationships

def semantic_accuracy_test(generated_sigs: List[str]) -> float:
    correct_relationships = 0
    total_relationships = 0
    
    for sig in generated_sigs:
        preprocessed_sig = preprocess_output(sig)
        lines = preprocessed_sig.split(" | ")
        
        for line in lines:
            main_category, generated_relationships = parse_semantics(line)
            if main_category in reference_set:
                reference_relationships = reference_set[main_category]
                
                for relationship in generated_relationships:
                    if relationship in reference_relationships:
                        correct_relationships += 1
                    total_relationships += 1
    
    accuracy = (correct_relationships / total_relationships) if total_relationships > 0 else 0
    print(f"Semantic Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# Minimal Prompt with Single Example for Flexibility
sig_prompt_with_single_example = """
You should add SIG example.
"""

# Generate multiple SIGs with single example-based prompt
generated_sigs_with_single_example = generate_sig_with_example(sig_prompt_with_single_example)

# Print generated SIG examples and validate Syntactic and Semantic Accuracy
print("Generated SIG Examples with Single Example-Based Prompt:")
syntactic_accuracy_test(generated_sigs_with_single_example)
semantic_accuracy_test(generated_sigs_with_single_example)

