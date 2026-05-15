---
title: MedAssist
emoji: 🩺
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# 🩺 MedAssist — AI Medical Information Chatbot

MedAssist is an AI-powered medical information chatbot built using a Retrieval-Augmented Generation (RAG) architecture. It is trained on 20,000 real doctor-patient consultations and provides grounded, contextually relevant health information through a clean web interface.

⚠️ Disclaimer: This chatbot is for general informational purposes only and does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional.

---

## 🌐 Live Demo
👉 https://huggingface.co/spaces/Wasim5647/MedAssist_Langchain

---

## 📌 Overview

MedAssist addresses the problem of hallucination in medical chatbots by grounding every response in real doctor-patient conversations retrieved at query time using semantic search. Instead of fine-tuning a model on medical data, MedAssist keeps a frozen general-purpose LLM and retrieves relevant medical context from a FAISS vector store built on the ChatDoctor-HealthCareMagic dataset.

---

## ⚙️ Tech Stack

Component            | Technology
---------------------|--------------------------------------------
RAG Framework        | LangChain
Vector Store         | FAISS
Embeddings           | sentence-transformers/all-MiniLM-L6-v2
LLM                  | Qwen2.5-7B-Instruct (HuggingFace Inference API)
Dataset              | ChatDoctor-HealthCareMagic-100k
Backend              | Flask
Deployment           | HuggingFace Spaces (Docker)
Development          | Google Colab

---

## ✨ Features

- RAG-based responses — every answer grounded in real doctor-patient consultations
- Semantic search — FAISS with MMR retrieval finds the most relevant and diverse context
- Multi-turn memory — remembers the last 4 conversation turns for follow-up questions
- Off-topic filtering — redirects non-medical queries back to health topics
- Emergency detection — identifies critical keywords and returns emergency service contacts
- Prompt engineering — strict rules prevent hallucination and symptom assumption
- Session reset — users can clear conversation history and start fresh
- Medical disclaimer — automatically appended to every response

---

## 🗂️ Project Structure

medassist-chatbot/
├── app.py              # Full self-contained Flask app with RAG pipeline
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container configuration for HuggingFace Spaces
└── README.md           # Project documentation

---

## 🚀 How to Run Locally

git clone https://github.com/Wasim5647/medassist-chatbot.git
cd medassist-chatbot
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token_here"
python app.py

Open http://localhost:7860 in your browser.

Note: First run takes around 10 minutes to download the dataset and build the FAISS index.

---

## 📊 Dataset

ChatDoctor-HealthCareMagic-100k — 100,000 real patient-doctor consultations sourced from HealthCareMagic. We use 20,000 samples for a balance between retrieval quality and startup speed.

Source: https://huggingface.co/datasets/lavita/ChatDoctor-HealthCareMagic-100k

---

## 👤 Author
Ahmad Wasim Wardak — MSc Applied AI, London South Bank University
