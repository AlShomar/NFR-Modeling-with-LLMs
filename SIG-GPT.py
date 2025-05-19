import sqlite3
import openai
import faiss
import numpy as np
import re
from flask import Flask, request, render_template
import os

# Set your OpenAI API key
openai.api_key = ""


# Path to the RAG database (pdf_chunks.db)
db_path = '/Users/ahmadalshomar/Desktop/PIG:SIG RAG/pdf_chunks.db'

# Example grammar definition for SIG generation
grammar_definition = """
Add the SIG Grammar Here
"""

# Function to query the RAG database and limit the number of retrieved chunks
def query_rag_db(query, max_chunks=3):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT chunk FROM pdf_chunks WHERE chunk LIKE ?", ('%' + query + '%',))
    results = cursor.fetchall()
    conn.close()
    return results[:max_chunks]  # Limit the number of retrieved chunks

# Function to truncate text to fit within token limits
def truncate_text(text, max_tokens=300):
    tokens = text.split()  # Simple word-based split
    return ' '.join(tokens[:max_tokens])  # Truncate to max_tokens

# Function to augment knowledge with RAG retrieval and generate SIG
def generate_sig_with_rag(prompt, query, max_chunks=3, max_output_tokens=300):
    # Retrieve relevant chunks from the RAG database (limit to max_chunks)
    relevant_chunks = query_rag_db(query, max_chunks=max_chunks)
    retrieved_text = "\n".join([chunk[0] for chunk in relevant_chunks])

    # Optionally truncate the retrieved text to fit within token limits
    retrieved_text = truncate_text(retrieved_text, max_tokens=500)

    # Combine the grammar definition with the truncated retrieved text
    augmented_prompt = f"""
    {grammar_definition}
    
    Retrieved Knowledge:
    {retrieved_text}
    
    Based on the above, generate a syntactically correct and semantically relevant Softgoal Interdependency Graph (SIG):
    {prompt}
    """
    
    # Use the ChatCompletion endpoint for GPT-4
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates Softgoal Interdependency Graphs."},
            {"role": "user", "content": augmented_prompt}
        ],
        max_tokens=max_output_tokens  # Adjust output size
    )
    
    return response['choices'][0]['message']['content'].strip()

# --- Evaluation Functions ---

# Grammar to check for syntactic correctness
grammar = {
    "CompositeProposition": r"[A-Za-z\s]+ : [A-Za-z\s]+( AND| OR| HELP| HURT) [A-Za-z\s]+",  # Matches CompositeProposition structure
    "Relation": r"(AND|OR|HELP|HURT)",  # Matches any valid relation
    "Proposition": r"[A-Za-z\s]+"  # Matches a simple string (proposition)
}

# Function to evaluate syntactic accuracy
def evaluate_syntactic_accuracy(sig, grammar):
    # Check if the SIG contains a basic structure matching CompositeProposition
    return bool(re.search(grammar["CompositeProposition"], sig))

# Function to evaluate semantic accuracy by checking for presence of key softgoals and relationships
def evaluate_semantic_accuracy(sig, intended_relationships):
    sig_lower = sig.lower()  # Lowercase SIG content for flexible matching
    
    # Check if each intended relationship (softgoal and relation) exists in the generated SIG
    for softgoal, relation in intended_relationships:
        softgoal_lower = softgoal.lower()
        relation_lower = relation.lower()
        
        # Ensure that both the softgoal and relationship pattern exist as substrings in the SIG
        if softgoal_lower not in sig_lower or relation_lower not in sig_lower:
            return False  # If any intended relationship is missing, return False
    
    return True

# Function to calculate syntactic accuracy as a percentage
def calculate_syntactic_accuracy(generated_sig, grammar):
    # Since we're evaluating one SIG at a time, total is 1
    correct = 1 if evaluate_syntactic_accuracy(generated_sig, grammar) else 0
    return (correct / 1) * 100  # Return percentage

# Function to calculate semantic accuracy as a percentage
def calculate_semantic_accuracy(generated_sig, intended_relationships):
    # Since we're evaluating one SIG at a time, total is 1
    correct = 1 if evaluate_semantic_accuracy(generated_sig, intended_relationships) else 0
    return (correct / 1) * 100  # Return percentage

# Example usage
query = "Security"  # Example keyword to retrieve knowledge from RAG
prompt = "Generate 10 SIGs for system security involving Confidentiality, Integrity, and Availability"

# Call the function to generate the SIG
generated_sig = generate_sig_with_rag(prompt, query)

# Print the generated SIG
print("Generated SIG:")
print(generated_sig)

# Intended relationships for the semantic check, focusing on key concepts and relations
intended_relationships = [
    ("Security", "Confidentiality AND Integrity AND Availability"),
    ("Confidentiality", "ImplementEncryption HELP Confidentiality"),
    ("Integrity", "IntegrityChecking AND RegularAuditing AND BackupData"),
    ("Availability", "ImplementBackupSystem HELP Availability")
]

# Calculate syntactic and semantic accuracy as percentages
syntactic_accuracy = calculate_syntactic_accuracy(generated_sig, grammar)
semantic_accuracy = calculate_semantic_accuracy(generated_sig, intended_relationships)

# Print the evaluation results
print("\nEvaluation Results:")
print(f"Syntactic Accuracy: {syntactic_accuracy}%")
print(f"Semantic Accuracy: {semantic_accuracy}%")

