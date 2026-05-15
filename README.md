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

MedAssist is an AI-powered medical information assistant developed using a Retrieval-Augmented Generation (RAG) architecture. The system is designed to answer general health-related questions by retrieving relevant doctor-patient consultation examples from a medical knowledge base and then generating a clear, context-aware response using an instruction-tuned large language model.

The project uses the `lavita/ChatDoctor-HealthCareMagic-100k` dataset, which contains real-world style patient questions and doctor responses. In this implementation, 20,000 consultation records are loaded from the dataset and converted into LangChain `Document` objects. Each document combines the patient’s question and the doctor’s response, allowing the system to retrieve medically relevant examples when a user asks a new question.

To support semantic search, the project uses the `sentence-transformers/all-MiniLM-L6-v2` embedding model through LangChain’s Hugging Face integration. These embeddings are stored in a FAISS vector database, enabling fast similarity search across the medical consultation corpus. The retriever uses Maximal Marginal Relevance (MMR), returning the top 5 relevant and diverse results from 20 candidate matches. This helps the chatbot avoid relying on near-duplicate examples and provides richer context to the language model.

For response generation, MedAssist uses `Qwen/Qwen2.5-7B-Instruct` through the Hugging Face Inference API. The retrieved medical examples are inserted into a carefully designed prompt template that instructs the model to treat them as reference material only, not as the current user’s personal medical history. The prompt also includes safety rules to prevent the chatbot from assuming symptoms, inventing personal details, repeating disclaimers unnecessarily, or giving overly personalised medical advice.

The chatbot maintains a short conversation history, using the most recent turns to support more natural follow-up interactions. A medical disclaimer is added to responses to remind users that the system is for general informational purposes only and should not replace advice from a qualified healthcare professional.

The application is implemented with Flask and includes a custom web interface for user interaction. The interface provides a clean chatbot layout, suggested health-related question buttons, a reset option, query counter, and responsive styling. The project also includes Docker support, making it suitable for deployment on Hugging Face Spaces using the Docker SDK.

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
