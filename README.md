---
title: MedAssist
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

## Project Description

MedAssist is an AI-powered medical information assistant designed to answer general health-related questions using a Retrieval-Augmented Generation (RAG) pipeline. The system combines LangChain, FAISS vector search, Hugging Face embeddings, and an instruction-tuned language model to retrieve relevant medical consultation examples before generating a response.

The chatbot uses the ChatDoctor-HealthCareMagic dataset as its knowledge source and retrieves similar doctor-patient examples to provide context-aware answers. It is deployed as a web application using Flask and Docker, making it suitable for hosting on Hugging Face Spaces.

MedAssist is designed for educational and informational use only. It does not provide diagnosis, treatment decisions, or personalised medical advice. Users are reminded to consult a qualified healthcare professional for any personal or urgent medical concerns.

## Key Features

- Retrieval-Augmented Generation pipeline using LangChain and FAISS
- Medical question-answering based on the ChatDoctor-HealthCareMagic dataset
- Hugging Face sentence-transformer embeddings for semantic search
- Instruction-tuned LLM response generation through Hugging Face Inference API
- Flask-based web interface with chatbot interaction
- Docker support for Hugging Face Spaces deployment
- Built-in medical disclaimer and health-focused response restrictions

## Technology Stack

- Python
- Flask
- LangChain
- FAISS
- Hugging Face Transformers / Inference API
- Sentence Transformers
- ChatDoctor-HealthCareMagic Dataset
- Docker
