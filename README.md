# 🧠 NFR Modeling with Large Language Models (LLMs)

This repository contains the **replication package** for the paper:

> ### 🎓 *Teaching LLMs Non-Functional Requirements Modeling: A Grammar and RAG Approach*  
> Submitted to **IEEE SSE 2025** (International Conference on Software Services Engineering)

---

## 📚 Overview

This project focuses on automating **Non-Functional Requirements (NFR)** modeling using **Large Language Models (LLMs)**.  
It integrates:

- ✅ **Text-based grammar** for syntactic control  
- ✅ **RAG (Retrieval-Augmented Generation)** to provide domain context  
- ✅ **Ontology-based reasoning** for softgoal classification  
- ✅ **SIG generation** using GPT-4 with minimal examples

---

## ⚙️ Model Configuration

| Parameter        | Value                    |
|------------------|--------------------------|
| **Model**        | `gpt-4` (OpenAI)         |
| **Temperature**  | `0.2`                    |
| **Max Tokens**   | `1024`                   |
| **Repetitions**  | `3` runs per prompt      |
| **Seed**         | Fixed (for reproducibility) |

> 💡 *All model parameters are defined in the codebase and can be adjusted in the `.py` files.*

---

## 📂 Components

- `SIG-GPT.py` – Web interface using Flask  
- `RAGNOGram.py` – SIG generation with RAG and examples  
- `NoRAGNOGram.py` – SIG generation without RAG (baseline)  
- `sig_generator.py` – Advanced SIG generator with ontology and grammar  
- `sig_ontology.json` – Ontology for softgoals and relationships  
- `GrammarUsed.txt` – Grammar specification for SIG syntax  
- `requirements.txt` – Python dependencies  
- `pdf_chunks.db` – **(external link or local)** knowledge base for RAG

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/yourname/NFR-Modeling-with-LLMs.git
cd NFR-Modeling-with-LLMs

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
