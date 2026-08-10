# PDF RAG — Retrieval-Augmented Generation

A beginner-friendly **Retrieval-Augmented Generation (RAG)** system that allows users to upload a PDF, retrieve relevant information from it, and generate answers based on the retrieved context.

This project is built to understand the **core concepts behind RAG** without relying on high-level frameworks such as LangChain or LlamaIndex.

---

## Overview

Large Language Models (LLMs) do not automatically have access to private documents.

RAG solves this problem by retrieving relevant information from an external knowledge source and providing it to the LLM as context.

The system follows this pipeline:

```text
PDF Document
     |
     v
Text Extraction
     |
     v
Text Chunking
     |
     v
Generate Embeddings
     |
     v
Store Embeddings
     |
     |
     +------------------+
     |                  |
     v                  v
User Question      Question Embedding
                        |
                        v
                 Similarity Search
                        |
                        v
                  Top-K Chunks
                        |
                        v
                Retrieved Context
                        |
                        v
              Context + Question
                        |
                        v
                    FLAN-T5
                        |
                        v
                   Final Answer
```

---

## Features

* PDF document ingestion
* Text extraction from PDF
* Custom text chunking
* Overlapping chunks
* Semantic text embeddings
* Cosine similarity search
* Top-K document retrieval
* Context-aware answer generation
* Open-source embedding model
* Open-source language model
* No LangChain dependency
* No LlamaIndex dependency
* Can run in GitHub Codespaces
* Can run locally
* Can run in Google Colab

---

## Tech Stack

| Technology                | Purpose                   |
| ------------------------- | ------------------------- |
| Python                    | Core programming language |
| PyPDF                     | PDF text extraction       |
| Sentence Transformers     | Text embeddings           |
| Scikit-learn              | Similarity calculation    |
| NumPy                     | Numerical operations      |
| Hugging Face Transformers | LLM inference             |
| FLAN-T5                   | Answer generation         |

---

## Models

### Embedding Model

```text
sentence-transformers/all-MiniLM-L6-v2
```

This model converts text into 384-dimensional vectors.

Example:

```text
"Machine learning is a branch of AI"
                  |
                  v
        Embedding Model
                  |
                  v
[0.12, 0.83, 0.41, ..., 0.27]
```

### Generation Model

```text
google/flan-t5-base
```

The retrieved document context and user question are provided to FLAN-T5 to generate the final answer.

---

# Project Structure

```text
PDF-RAG/
│
├── data/
│   └── documents/
│
├── src/
│   ├── loader.py
│   ├── chunker.py
│   ├── embeddings.py
│   ├── retriever.py
│   └── generator.py
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Running in GitHub Codespaces

## 1. Clone the Repository

If the repository is already open in Codespaces, skip this step.

```bash
git clone <your-repository-url>
cd PDF-RAG
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

### Linux / GitHub Codespaces

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist yet:

```bash
pip install pypdf sentence-transformers scikit-learn transformers accelerate
```

Then create:

```text
requirements.txt
```

with:

```text
pypdf
sentence-transformers
scikit-learn
numpy
transformers
accelerate
torch
```

---

# How the RAG Works

## 1. Document Loading

The PDF is loaded using PyPDF.

```python
from pypdf import PdfReader

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"
```

The PDF is converted into plain text.

---

## 2. Text Chunking

Large documents are split into smaller pieces.

For example:

```text
Document
   |
   +-- Chunk 1
   +-- Chunk 2
   +-- Chunk 3
   +-- Chunk 4
   +-- ...
```

The project uses overlapping chunks.

```python
def create_chunks(text, chunk_size=500, overlap=100):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks
```

The overlap helps preserve information between neighboring chunks.

---

# 3. Generate Embeddings

Each chunk is converted into a numerical vector.

```python
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

chunk_embeddings = embedding_model.encode(chunks)
```

Example:

```text
Chunk
 |
 v
Embedding Model
 |
 v
384-dimensional vector
```

---

# 4. Embed the User Query

The user's question is also converted into an embedding.

```python
question_embedding = embedding_model.encode(
    [question]
)
```

Now we have:

```text
Document Chunks
      |
      v
Chunk Embeddings

User Question
      |
      v
Question Embedding
```

---

# 5. Similarity Search

The question vector is compared against the document vectors.

The project uses cosine similarity.

```python
from sklearn.metrics.pairwise import cosine_similarity

similarities = cosine_similarity(
    question_embedding,
    chunk_embeddings
)[0]
```

Example:

```text
Chunk                    Score
--------------------------------
Machine Learning          0.91
Neural Networks           0.84
Databases                 0.32
Computer Networks         0.18
```

Higher similarity means the chunk is more relevant to the question.

---

# 6. Retrieve Top-K Chunks

The system retrieves the most relevant chunks.

```python
import numpy as np

top_k = 3

top_indices = np.argsort(
    similarities
)[-top_k:][::-1]
```

For example:

```text
Question
   |
   v
Similarity Search
   |
   +-- Chunk 17  -> 0.91
   +-- Chunk 23  -> 0.87
   +-- Chunk 42  -> 0.83
```

These chunks become the context for the LLM.

---

# 7. Build the Context

```python
context = "\n\n".join(
    chunks[index]
    for index in top_indices
)
```

The system now has:

```text
Retrieved Context
       +
User Question
```

---

# 8. Generate the Answer

Load FLAN-T5:

```python
from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)
```

Create the prompt:

```python
prompt = f"""
Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
```

Generate:

```python
result = generator(
    prompt,
    max_new_tokens=200
)

answer = result[0]["generated_text"]

print(answer)
```

---

# RAG Architecture

The project contains two major stages.

## Retrieval

```text
Question
    |
    v
Question Embedding
    |
    v
Similarity Search
    |
    v
Relevant Chunks
```

## Generation

```text
Relevant Chunks
       +
Question
       |
       v
     Prompt
       |
       v
      LLM
       |
       v
    Answer
```

Therefore:

```text
RAG
=
Retrieval
+
Augmented Context
+
Generation
```

---

# Example

Suppose the PDF contains information about machine learning.

The user asks:

```text
What is overfitting?
```

The system performs:

```text
Question
   |
   v
Question Embedding
   |
   v
Search PDF Chunks
   |
   v
Top Relevant Chunks
   |
   v
"Overfitting occurs when..."
   |
   v
FLAN-T5
   |
   v
Generated Answer
```

The LLM receives the retrieved information instead of relying only on its pretrained knowledge.

---

# Current Limitations

This is an educational implementation.

Current limitations include:

* Embeddings are stored in memory
* No persistent vector database
* Basic character-based chunking
* No metadata filtering
* No reranking
* No conversation memory
* Limited document format support
* No OCR for scanned PDFs
* Small generation model
* No source/page citations

---

# Future Improvements

## Vector Database

Replace in-memory embeddings with a vector database such as:

```text
FAISS
ChromaDB
Qdrant
Pinecone
```

---

## Better Chunking

Implement:

* Sentence-based chunking
* Paragraph-based chunking
* Recursive chunking
* Semantic chunking

---

## Multiple Documents

Allow the system to create a knowledge base from multiple PDFs.

```text
PDF 1
PDF 2
PDF 3
PDF 4
 |
 v
Document Processing
 |
 v
Embeddings
 |
 v
Vector Database
 |
 v
RAG
```

---

## Reranking

Improve retrieval quality by adding a reranker.

```text
Question
   |
   v
Vector Search
   |
   v
Top 20 Chunks
   |
   v
Reranker
   |
   v
Top 5 Chunks
   |
   v
LLM
```

---

## Conversational RAG

Add conversation memory so users can ask follow-up questions.

```text
User Question
      |
      v
Conversation History
      |
      v
Retriever
      |
      v
Relevant Context
      |
      v
LLM
      |
      v
Answer
```

---

## Source Citations

The system can be extended to return the source document and page number.

Example:

```text
Answer:
Supervised learning uses labeled training data.

Sources:
- machine_learning.pdf
- Page 12
- Page 14
```

---

# Future Production Architecture

```text
                    ┌───────────────┐
                    │    Frontend   │
                    │ React / Next  │
                    └───────┬───────┘
                            |
                            v
                    ┌───────────────┐
                    │    FastAPI    │
                    └───────┬───────┘
                            |
              ┌─────────────┴─────────────┐
              |                           |
              v                           v
       Document Pipeline            Query Pipeline
              |                           |
              v                           v
          Chunking                 Query Embedding
              |                           |
              v                           v
         Embeddings                 Vector Search
              |                           |
              v                           v
       Vector Database              Top-K Results
              |                           |
              └─────────────┬─────────────┘
                            |
                            v
                         Reranker
                            |
                            v
                     Context Builder
                            |
                            v
                           LLM
                            |
                            v
                      Final Response
```

---

# Learning Objectives

This project helps understand:

* Retrieval-Augmented Generation
* Document ingestion
* PDF processing
* Text chunking
* Embeddings
* Vector representations
* Cosine similarity
* Semantic search
* Top-K retrieval
* Context augmentation
* Prompt construction
* LLM generation
* RAG architecture

---

# Why Build RAG From Scratch?

Frameworks such as LangChain make RAG development much faster, but they can hide what happens internally.

This project intentionally implements the important steps manually:

```text
PDF
 |
 v
Chunking
 |
 v
Embedding
 |
 v
Similarity
 |
 v
Retrieval
 |
 v
Context
 |
 v
LLM
 |
 v
Answer
```

After understanding this implementation, frameworks such as LangChain, LlamaIndex, and vector databases become much easier to learn.

---

# Roadmap

* [x] PDF text extraction
* [x] Text chunking
* [x] Overlapping chunks
* [x] Embedding generation
* [x] Cosine similarity
* [x] Top-K retrieval
* [x] Context construction
* [x] LLM generation
* [ ] Persistent vector database
* [ ] Multiple PDF support
* [ ] Better chunking
* [ ] Reranking
* [ ] Source citations
* [ ] Conversational memory
* [ ] FastAPI backend
* [ ] Web interface
* [ ] Production deployment

---

# Author

**Sahil**

BTech — Artificial Intelligence & Machine Learning

---

# License

This project is intended for educational and research purposes.
