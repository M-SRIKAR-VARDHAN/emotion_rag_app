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
from langchain_huggingface import HuggingFaceEmbeddings
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
import time # For debug timing

# --- Your Hugging Face Repos (from your screenshot) ---
EMOTION_MODEL_REPO = "primal-sage/emotion-model" 
RAG_FILES_REPO = "primal-sage/my-rag-index" 
# --- No more changes needed here ---

# --- Local directory to store downloaded index ---
FAISS_INDEX_DIR = "faiss_index_local"

print("DEBUG: model_utils.py file loaded by Python.")


@st.cache_resource
def load_emotion_model():
    """Loads the fine-tuned emotion classification model and processor."""
    print("DEBUG: --- load_emotion_model START ---")
    st.write("Cache miss: Loading emotion model...")
    
    print("DEBUG: Loading processor from:", EMOTION_MODEL_REPO)
    processor = AutoImageProcessor.from_pretrained(EMOTION_MODEL_REPO)
    print("DEBUG: Loading model from:", EMOTION_MODEL_REPO)
    model = AutoModelForImageClassification.from_pretrained(EMOTION_MODEL_REPO)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEBUG: Setting emotion model device to: {device}")
    model.to(device)
    
    print("DEBUG: --- load_emotion_model END ---")
    return processor, model, device

# Copied from your AI_part2, Cell 6
PROMPT_TEMPLATE = """Based on the customer reviews below, provide a clear and concise summary that directly answers the question.
Customer Reviews:
{context}
Question: {question}
Provide a brief summary (2-3 sentences maximum):"""

# Copied from your AI_part2, Cell 6
class DiverseRetriever(BaseRetriever):
    print("DEBUG: Class DiverseRetriever is being defined.")
    vectorstore: object
    k: int = 4
    
    # --- THIS IS THE FIX (Pydantic v2 syntax) ---
    model_config = {"arbitrary_types_allowed": True}
    # --- End of Fix ---

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        print(f"DEBUG: DiverseRetriever running _get_relevant_documents for query: {query}")
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
        print(f"DEBUG: DiverseRetriever found {len(selected)} documents.")
        return selected[:self.k]

@st.cache_resource
def load_rag_pipeline():
    """Loads all components for the RAG and Sentiment pipelines."""
    print("DEBUG: --- load_rag_pipeline START ---")
    st.write("Cache miss: Loading RAG & Sentiment pipelines...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_id = 0 if device == "cuda" else -1
    print(f"DEBUG: RAG device: {device} (device_id: {device_id})")

    print("DEBUG: Loading HuggingFaceEmbeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("DEBUG: Embeddings loaded.")

    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    print(f"DEBUG: Downloading RAG files from {RAG_FILES_REPO} to {FAISS_INDEX_DIR} (FORCING RE-DOWNLOAD)")
    
    # --- START OF THE NEW FIX ---
    hf_hub_download(
        repo_id=RAG_FILES_REPO, 
        filename="index.faiss", 
        local_dir=FAISS_INDEX_DIR, 
        force_download=True  # <-- Forces download of new file
    )
    hf_hub_download(
        repo_id=RAG_FILES_REPO, 
        filename="index.pkl", 
        local_dir=FAISS_INDEX_DIR, 
        force_download=True  # <-- Forces download of new file
    )
    # --- END OF THE NEW FIX ---
    
    print("DEBUG: RAG files downloaded.")

    print("DEBUG: Loading FAISS index...")
    db = FAISS.load_local(
        FAISS_INDEX_DIR, 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    print("DEBUG: FAISS index loaded.")

    # THIS IS THE LINE THAT WAS CAUSING THE 'fields_set' ERROR
    print("DEBUG: Initializing DiverseRetriever...")
    diverse_retriever = DiverseRetriever(vectorstore=db, k=4)
    print("DEBUG: DiverseRetriever initialized.") # If you see this, the error is fixed.

    print("DEBUG: Loading 'text2text-generation' pipeline (flan-t5-base)...")
    llm_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_length=150,
        device=device_id
    )
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    print("DEBUG: LLM pipeline loaded.")

    PROMPT = PromptTemplate(template=PROMPT_TEMPLATE, input_variables=["context", "question"])
    
    print("DEBUG: Initializing RetrievalQA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=diverse_retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("DEBUG: RetrievalQA chain initialized.")

    print("DEBUG: Loading 'sentiment-analysis' pipeline...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=device_id
    )
    sentiment_labels = {"LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE"}
    print("DEBUG: Sentiment pipeline loaded.")
    
    print("DEBUG: --- load_rag_pipeline END ---")
    return qa_chain, sentiment_pipe, sentiment_labels

def predict_emotion(processor, model, device, image: Image.Image):
    """Predicts the emotion from a PIL Image."""
    print("DEBUG: --- predict_emotion START ---")
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        
    predicted_class_id = logits.argmax(-1).item()
    emotion = model.config.id2label[predicted_class_id]
    print(f"DEBUG: Emotion predicted: {emotion}")
    return emotion

def query_rag(qa_chain, query):
    """Runs the RAG query and returns the result."""
    print(f"DEBUG: --- query_rag START --- (Query: {query})")
    result = qa_chain({"query": query})
    print("DEBUG: RAG query finished.")
    return result

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