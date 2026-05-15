---
title: MedAssist
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🩺 MedAssist — AI Medical Information Assistant

## Project Description

### Overview

MedAssist is an AI-powered medical information assistant developed using a Retrieval-Augmented Generation (RAG) architecture. The system is designed to answer general health-related questions by retrieving relevant doctor-patient consultation examples from a medical knowledge base and then generating a clear, context-aware response using an instruction-tuned large language model.

### Knowledge Base

The project uses the `lavita/ChatDoctor-HealthCareMagic-100k` dataset, which contains real-world style patient questions and doctor responses. In this implementation, 20,000 consultation records are loaded from the dataset and converted into LangChain `Document` objects.

Each document combines the patient’s question and the doctor’s response, allowing the system to retrieve medically relevant examples when a user asks a new question.

### Retrieval-Augmented Generation Pipeline

MedAssist uses a RAG pipeline to improve the quality and relevance of its answers. Instead of relying only on the language model’s general knowledge, the system first searches the medical consultation database for similar examples.

The retrieved examples are then passed to the language model as supporting context. This helps the chatbot produce more grounded and context-aware responses.

### Embeddings and Vector Search

To support semantic search, the project uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model through LangChain’s Hugging Face integration.

The generated embeddings are stored in a FAISS vector database, which enables fast similarity search across the medical consultation corpus.

### Retrieval Strategy

The retriever uses Maximal Marginal Relevance (MMR), returning the top 5 relevant and diverse results from 20 candidate matches.

This helps reduce duplicate or overly similar retrieved examples and provides richer context to the language model before response generation.

### Language Model

For response generation, MedAssist uses `Qwen/Qwen2.5-7B-Instruct` through the Hugging Face Inference API.

The retrieved medical examples are inserted into a structured prompt template, which guides the model to generate helpful answers based on the retrieved context.

### Prompt Design and Safety

The prompt is designed to make the chatbot treat retrieved examples as reference material only, not as the current user’s personal medical history.

It also includes safety instructions to prevent the chatbot from assuming symptoms, inventing personal details, repeating disclaimers unnecessarily, or giving overly personalised medical advice.

### Conversation Memory

The chatbot maintains a short conversation history using the most recent turns. This allows the system to handle simple follow-up questions more naturally while keeping the interaction focused and lightweight.

### Web Application

The application is implemented using Flask and includes a custom web interface for user interaction.

The interface provides a clean chatbot layout, suggested health-related question buttons, a reset option, query counter, and responsive styling.

### Deployment

The project includes Docker support, making it suitable for deployment on Hugging Face Spaces using the Docker SDK.

This allows the chatbot to run as a web-based application without requiring users to install the project locally.

### Intended Use

MedAssist is designed for educational and informational purposes only. It does not provide medical diagnosis, treatment decisions, prescriptions, or personalised medical advice.

Users should always consult a qualified healthcare professional for personal medical concerns, urgent symptoms, or treatment decisions.

## Key Features

- Retrieval-Augmented Generation medical chatbot
- LangChain-based document processing and prompt management
- FAISS vector database for fast semantic search
- MMR retrieval for relevant and diverse medical context
- Hugging Face `all-MiniLM-L6-v2` sentence embeddings
- Qwen2.5-7B-Instruct model for answer generation
- Flask-based chatbot web interface
- Docker-based deployment for Hugging Face Spaces
- Short conversation memory for follow-up questions
- Built-in medical disclaimer and safety-focused prompting

## How the System Works

1. The ChatDoctor-HealthCareMagic dataset is loaded from Hugging Face.
2. Patient questions and doctor responses are converted into LangChain documents.
3. The documents are embedded using a Hugging Face sentence-transformer model.
4. The embeddings are stored in a FAISS vector database.
5. When a user asks a question, the retriever finds the most relevant medical consultation examples.
6. The retrieved examples are inserted into a structured prompt.
7. Qwen2.5-7B-Instruct generates a response based on the retrieved context.
8. The Flask web app displays the answer to the user with a medical disclaimer.

## Technology Stack

- Python
- LangChain
- FAISS
- Hugging Face Datasets
- Hugging Face Inference API
- Sentence Transformers
- Qwen2.5-7B-Instruct
- Flask
- Flask-CORS
- Docker
- Hugging Face Spaces

## Medical Disclaimer

MedAssist is designed for educational and informational purposes only. It does not provide medical diagnosis, treatment decisions, prescriptions, or personalised medical advice. Users should always consult a qualified healthcare professional for personal medical concerns, urgent symptoms, or treatment decisions.
