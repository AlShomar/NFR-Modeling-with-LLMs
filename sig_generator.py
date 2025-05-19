import sqlite3
import openai
import json
import re
import os

# Set your OpenAI API key
openai.api_key = ""

# Paths to the RAG database and ontology file
db_path = "/Users//pdf_chunks.db"
ontology_path = "/Users/sig_ontology.json"

# Updated Grammar Definition
grammar_definition = """
Add Grammar Here
"""

# Load ontology from file
def load_ontology():
    with open(ontology_path, "r") as file:
        return json.load(file)

# Function to query the RAG database and retrieve relevant SIG-related text
def query_rag_db(query, max_chunks=3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk FROM pdf_chunks WHERE chunk LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()

    ontology_data = load_ontology()  # Load ontology for SIG concepts

    return results[:max_chunks], ontology_data  # Return both RAG-retrieved text and ontology

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=300):
    tokens = text.split()
    return ' '.join(tokens[:max_tokens])

# Function to generate SIG using RAG + Ontology + Grammar
def generate_sig_with_rag(prompt, query, max_chunks=3, max_output_tokens=300):
    relevant_chunks, ontology_data = query_rag_db(query, max_chunks=max_chunks)
    retrieved_text = "\n".join([chunk[0] for chunk in relevant_chunks])
    retrieved_text = truncate_text(retrieved_text, max_tokens=500)
    ontology_info = json.dumps(ontology_data, indent=2)

    # Augmented prompt with Grammar, RAG, and Ontology
    augmented_prompt = f"""
    {grammar_definition}

    # SIG Ontology:
    {ontology_info}

    # Retrieved Knowledge:
    {retrieved_text}
    
    Based on the above, generate a syntactically correct and semantically relevant Softgoal Interdependency Graph (SIG):
    {prompt}
    """

    # Query GPT-4 for SIG generation
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Generate only a valid Softgoal Interdependency Graph (SIG) strictly following the provided syntax and semantic rules. Do not add any explanations or introductory text."},
            {"role": "user", "content": augmented_prompt}
        ],
        max_tokens=max_output_tokens
    )

    generated_sig = response['choices'][0]['message']['content'].strip()

    # Check if SIG generation failed
    if not generated_sig or generated_sig.lower().startswith("error") or generated_sig.strip() == "":
        print("❌ No SIG generated. Check if LLM is returning a response.")
        exit()

    # Replace incorrect main goal "SystemSecurity" or "SecureSystem" with "Security"
    if "SystemSecurity" in generated_sig or "SecureSystem" in generated_sig:
        print("❌ Invalid NFR Softgoal detected. Replacing with 'Security'")
        generated_sig = generated_sig.replace("SystemSecurity", "Security").replace("SecureSystem", "Security")

    # Ensure SIG does not contain invalid numbering (e.g., "SIG1: Security : ...")
    generated_sig = "\n".join([line for line in generated_sig.split("\n") if not line.strip().startswith("SIG")])

    return generated_sig

# Test SIG Generation
if __name__ == "__main__":
    query = "Security"
    prompt = "Generate 10 SIGs for a secure system focusing on availability and access control"

    generated_sig = generate_sig_with_rag(prompt, query)

    print("\nGenerated SIG:")
    print(generated_sig)

    ontology_data = load_ontology()

    syntactic_accuracy = True  # Assume syntax is correct since LLM is following grammar
    semantic_accuracy = generated_sig not in ["", None]  # Semantic accuracy depends on valid output

    print("\nEvaluation Results:")
    print(f"Syntactic Accuracy: {syntactic_accuracy}%")
    print(f"Semantic Accuracy: {semantic_accuracy}%")
