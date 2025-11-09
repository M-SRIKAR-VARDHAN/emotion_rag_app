# model_utils.py
import streamlit as st
import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.schema.retriever import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from PIL import Image
import numpy as np
import os
from huggingface_hub import hf_hub_download

# --- Your Hugging Face Repos (from your screenshot) ---
EMOTION_MODEL_REPO = "primal-sage/emotion-model" 
RAG_FILES_REPO = "primal-sage/my-rag-index" 
# --- No more changes needed here ---

# --- Local directory to store downloaded index ---
FAISS_INDEX_DIR = "faiss_index_local"


@st.cache_resource
def load_emotion_model():
    """Loads the fine-tuned emotion classification model and processor."""
    st.write("Cache miss: Loading emotion model...")
    processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_REPO)
    model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_REPO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("Emotion model loaded.")
    return processor, model, device

# Copied from your AI_part2, Cell 6
PROMPT_TEMPLATE = """Based on the customer reviews below, provide a clear and concise summary that directly answers the question.
Customer Reviews:
{context}
Question: {question}
Provide a brief summary (2-3 sentences maximum):"""

# Copied from your AI_part2, Cell 6
class DiverseRetriever(BaseRetriever):
    vectorstore: object
    k: int = 4
    class Config:
        arbitrary_types_allowed = True
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        candidates = self.vectorstore.similarity_search(query, k=20)
        selected = []
        seen_texts = set()
        for doc in candidates:
            text_start = doc.page_content[:50]
            if text_start not in seen_texts and len(selected) < self.k:
                selected.append(doc)
                seen_texts.add(text_start)
        if len(selected) < self.k:
            for doc in candidates:
                if len(selected) >= self.k:
                    break
                if doc not in selected:
                    selected.append(doc)
        return selected[:self.k]

@st.cache_resource
def load_rag_pipeline():
    """Loads all components for the RAG and Sentiment pipelines."""
    st.write("Cache miss: Loading RAG & Sentiment pipelines...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    hf_hub_download(repo_id=RAG_FILES_REPO, filename="index.faiss", local_dir=FAISS_INDEX_DIR)
    hf_hub_download(repo_id=RAG_FILES_REPO, filename="index.pkl", local_dir=FAISS_INDEX_DIR)
    db = FAISS.load_local(
        FAISS_INDEX_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    diverse_retriever = DiverseRetriever(vectorstore=db, k=4)
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=150,
        device=device_id
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=diverse_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device_id
    )
    sentiment_labels = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    print("RAG and Sentiment pipelines loaded.")
    return qa_chain, sentiment_pipe, sentiment_labels

def predict_emotion(processor, model, device, image: Image.Image):
    """Predicts the emotion from a PIL Image."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    predicted_class_id = logits.argmax(-1).item()
    emotion = model.config.id2label[predicted_class_id]
    return emotion

def query_rag(qa_chain, query):
    """Runs the RAG query and returns the result."""
    return qa_chain({"query": query})

def analyze_sentiment(sentiment_pipe, labels_map, text):
    """Analyzes and formats the sentiment of a text."""
    try:
        sentiment = sentiment_pipe(text)[0]
        label = labels_map.get(sentiment['label'], 'UNKNOWN')
        score = sentiment['score']
        return f"{label} (Score: {score:.2f})"
    except Exception as e:
        print(f"Error in sentiment analysis: {e}")
        return "Error"