import openai
import sqlite3
from typing import List, Tuple, Dict


# Set your OpenAI API key
openai.api_key = ""

# Path to the RAG database (pdf_chunks.db)
db_path = '/Users/pdf_chunks.db'

# Allowed relations and reference set for semantic accuracy
allowed_relations = {"AND", "OR", "HELP", "HURT"}
reference_set = {
    "Security": [("Confidentiality", "AND", "Integrity"), ("Integrity", "AND", "Availability")],
    "Usability": [("AvoidRestrictiveAccess", "HURT", "Security")],
    "Performance": [("TimePerformance", "OR", "SpacePerformance")]
}

# Function to query the RAG database and limit the number of retrieved chunks
def query_rag_db(query, max_chunks=3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk FROM pdf_chunks WHERE chunk LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    return results[:max_chunks]

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=300):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens])

# Function to clean up the generated output by removing labels like "SIG 1:", "Based on the given NFR context,"
def clean_generated_output(generated_sig: str) -> str:
    lines = generated_sig.splitlines()
    filtered_lines = [
        line for line in lines if not line.startswith("SIG") and not line.startswith("Based on") and line.strip()
    ]
    return "\n".join(filtered_lines)

# Function to generate SIGs with example-based learning
def generate_sig_with_examples(prompt, query, max_chunks=3, max_output_tokens=300, n=5):
    generated_sigs = []
    for i in range(n):
        relevant_chunks = query_rag_db(query, max_chunks=max_chunks)
        retrieved_text = "\n".join([chunk[0] for chunk in relevant_chunks])
        retrieved_text = truncate_text(retrieved_text, max_tokens=500)
        
        # Augment the prompt with retrieved text and example SIGs
        augmented_prompt = f"{retrieved_text}\n\n{prompt}"
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant knowledgeable in SIG modeling."},
                      {"role": "user", "content": augmented_prompt}],
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

# Improved Prompt with Example SIGs but Flexible Instructions
sig_prompt_with_examples = """
You Should write SIG example. Better examples get better results.
"""

# Generate multiple SIGs with example-based learning prompt
generated_sigs_with_examples = generate_sig_with_examples(sig_prompt_with_examples, query="non-functional requirements")

# Print generated SIG examples and validate Syntactic and Semantic Accuracy
print("Generated SIG Examples with Learning from Examples:")
syntactic_accuracy_test(generated_sigs_with_examples)
semantic_accuracy_test(generated_sigs_with_examples)
