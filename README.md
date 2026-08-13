# RAG Algorithms Experimental Study

This project contains experimental implementations of multiple **Retrieval-Augmented Generation (RAG)** approaches used as the foundation for the **TRUST Index** project.

The purpose of this repository is to study how different RAG architectures retrieve information, generate responses, handle incorrect retrieval, and reduce hallucinations. We also experiment with different **Hugging Face embedding/encoder models** and **open-source LLMs**.

---

## Algorithms Studied

The following approaches are implemented and evaluated:

* Traditional RAG
* Corrective RAG (CRAG)
* Self-RAG (SRAG)
* GraphRAG (GRAG)
* ReAct

The generated responses from these systems can later be passed to the **TRUST Index verification layer**.

---

## 1. Traditional RAG

Traditional RAG follows a simple pipeline:

```text
User Query
    ↓
Query Embedding
    ↓
Vector Database
    ↓
Relevant Documents
    ↓
LLM
    ↓
Generated Answer
```

### Main Components

* Document processing
* Text chunking
* Sentence Transformers
* FAISS / ChromaDB
* Semantic similarity search
* Open-source LLM

Traditional RAG serves as the **baseline** for comparison with the other approaches.

---

## 2. Corrective RAG (CRAG)

Corrective RAG evaluates the quality of retrieved information before generating the final response.

```text
User Query
    ↓
Initial Retrieval
    ↓
Retrieval Evaluation
    ↓
 ┌───────────────┐
 │               │
Good          Poor/Ambiguous
 │               │
 │          Corrective Retrieval
 │               │
 └───────┬───────┘
         ↓
       LLM
         ↓
   Final Answer
```

### Key Idea

If the initial retrieval is not sufficiently relevant, the system performs additional retrieval to obtain better evidence.

### Purpose

CRAG helps reduce errors caused by poor retrieval quality.

---

## 3. Self-RAG (SRAG)

Self-RAG introduces a self-reflection mechanism where the system evaluates its generated response and determines whether additional retrieval or revision is required.

```text
User Query
    ↓
Retrieval
    ↓
LLM
    ↓
Initial Answer
    ↓
Self-Critique
    ↓
 ┌──────────────┐
 │              │
Supported    Needs Retrieval
 │              │
 │        Additional Retrieval
 │              │
 └───────┬──────┘
         ↓
    Final Answer
```

### Key Idea

The model can:

* Retrieve evidence
* Generate an answer
* Critique the answer
* Request additional information
* Revise the response

This makes Self-RAG more adaptive than traditional RAG.

---

## 4. GraphRAG (GRAG)

GraphRAG represents information and relationships between entities using a graph structure.

```text
Documents
    ↓
Entity / Relationship Extraction
    ↓
Knowledge Graph
    ↓
Graph Retrieval
    ↓
Relevant Context
    ↓
LLM
    ↓
Answer
```

### Graph Components

Nodes can represent:

* Entities
* Concepts
* Documents
* Claims

Edges represent relationships between those nodes.

### Advantage

Graph-based retrieval can be useful for questions requiring relationships between multiple pieces of information or multi-hop reasoning.

---

## 5. ReAct

ReAct combines **reasoning and actions**.

The system can decide to perform actions such as:

* Search
* Retrieve documents
* Query a knowledge base
* Use a tool

The general workflow is:

```text
User Query
    ↓
Reason
    ↓
Action
    ↓
Observation
    ↓
Reason
    ↓
Action
    ↓
Observation
    ↓
Final Answer
```

### Key Idea

Instead of generating an answer directly, the model can interact with external tools and use the obtained observations to construct its response.

---

# Model Experimentation

A major part of this project is comparing different combinations of **embedding models and LLMs**.

## Embedding / Encoder Models

We experiment with Hugging Face and Sentence Transformer models such as:

```text
all-MiniLM-L6-v2
all-mpnet-base-v2
Other compatible sentence-transformer models
```

The encoder converts text into numerical vectors:

```text
Text
 ↓
Encoder
 ↓
Embedding Vector
 ↓
Vector Database
```

These embeddings are used for semantic retrieval and similarity comparison.

---

# Open-Source LLM Experiments

Different open-source LLMs can be tested with the RAG architectures.

Example models include:

```text
Qwen
Llama
Mistral
Gemma
Phi
```

The exact model can be changed depending on available RAM, GPU memory, and inference requirements.

---

# Experimental Pipeline

Each RAG architecture is evaluated using different model combinations:

```text
                 ┌───────────────┐
                 │  RAG Methods  │
                 └───────┬───────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
      CRAG             SRAG             GRAG
        ↓                ↓                ↓
        └────────────────┼────────────────┘
                         ↓
                     ReAct
                         ↓
              Embedding Models
                         ↓
                   Vector Search
                         ↓
                  Open-Source LLM
                         ↓
                 Generated Response
```

---

# Evaluation

The different combinations are evaluated using factors such as:

### Retrieval Quality

* Relevance of retrieved documents
* Similarity scores
* Top-K retrieval quality

### Response Quality

* Factual correctness
* Relevance
* Completeness
* Context utilization

### Hallucination Analysis

* Supported information
* Unsupported information
* Contradictory information

### Performance

* Retrieval latency
* Generation latency
* Memory consumption
* Model size
* Computational requirements

---

# Experimental Comparison

The experiments can be organized as:

| RAG Method | Encoder | LLM   | Retrieval | Response Quality | Hallucination |
| ---------- | ------- | ----- | --------- | ---------------- | ------------- |
| RAG        | Model A | LLM A | —         | —                | —             |
| CRAG       | Model A | LLM A | —         | —                | —             |
| SRAG       | Model A | LLM A | —         | —                | —             |
| GraphRAG   | Model A | LLM A | —         | —                | —             |
| ReAct      | Model A | LLM A | —         | —                | —             |

The table can be populated after completing the experiments.

---

# Project Structure

```text
rag-experiments/
│
├── data/
│   ├── documents/
│   └── processed/
│
├── embeddings/
│
├── vector_db/
│
├── rag/
│   ├── traditional_rag.py
│   ├── corrective_rag.py
│   ├── self_rag.py
│   ├── graph_rag.py
│   └── react.py
│
├── models/
│
├── evaluation/
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

# Technologies Used

* Python
* Hugging Face Transformers
* Sentence Transformers
* FAISS
* ChromaDB
* LangChain
* NetworkX
* NumPy
* Pandas
* Open-source LLMs

---

# Relationship With TRUST Index

These experiments form the **baseline and experimental layer** of the TRUST Index project.

The final TRUST Index architecture is:

```text
                 User Query
                     ↓
        ┌────────────────────────┐
        │ RAG / Agentic Systems  │
        │                        │
        │ RAG                    │
        │ CRAG                   │
        │ Self-RAG               │
        │ GraphRAG               │
        │ ReAct                  │
        └───────────┬────────────┘
                    ↓
              Generated Answer
                    ↓
          ┌─────────────────────┐
          │    TRUST INDEX      │
          │                     │
          │ Claim Extraction    │
          │ Evidence Retrieval  │
          │ Semantic Similarity │
          │ NLI Verification    │
          │ Evidence Graph      │
          │ Trust Propagation   │
          └──────────┬──────────┘
                     ↓
                TRUST SCORE
                     ↓
        ┌────────────┼────────────┐
        ↓            ↓            ↓
     TRUSTED     UNCERTAIN   HALLUCINATED
```

The TRUST Index therefore does not need to replace existing RAG approaches. Instead, it acts as an **independent verification layer** that can evaluate responses generated by different RAG and agentic architectures. This matches the proposed architecture in the project synopsis.

---

# Objective

The primary objective of this experimental phase is to understand the behavior of different RAG architectures and identify suitable **embedding models, retrieval strategies, and open-source LLMs** for integration with the TRUST Index verification framework.

The experiments will provide the baseline against which the final TRUST Index system can be evaluated.

---

## Future Work

The next stage is to implement the TRUST verification engine:

```text
Generated Response
       ↓
Claim Extraction
       ↓
Evidence Retrieval
       ↓
Semantic Similarity
       ↓
NLI Contradiction Detection
       ↓
Evidence Graph
       ↓
Trust Propagation
       ↓
TRUST Score
       ↓
Explainable Result
```

This will allow the project to move from simply **generating RAG responses** to **systematically evaluating the reliability of those responses**.
