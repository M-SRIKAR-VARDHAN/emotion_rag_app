# streamlit_app.py
import streamlit as st
from PIL import Image
import model_utils  # Our "backend" file
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="Facial Emotion RAG System",
    page_icon="🤖",
    layout="wide"
)

st.title(" facial Emotion-Driven Review Analysis 🤖")
st.write("This app combines two AI systems. Upload a photo to see a facial emotion, and the system will automatically query a review database for that emotion.")

# --- Load Models ---
try:
    with st.spinner("Warming up the AI models... This may take a moment the first time."):
        start_time = time.time()
        emotion_processor, emotion_model, emotion_device = model_utils.load_emotion_model()
        qa_chain, sentiment_pipe, sentiment_labels = model_utils.load_rag_pipeline()
        end_time = time.time()
    
    st.success(f"AI Models are ready! (Loaded in {end_time - start_time:.2f}s)")

except Exception as e:
    st.error(f"Error loading models: {e}")
    st.error("Please check your Hugging Face repository names in `model_utils.py` and ensure they are public.")
    st.stop()


# --- App Layout ---
tab1, tab2 = st.tabs(["📸 Query with a Photo (BONUS TASK)", "💬 Manual Text Query"])

# --- TAB 1: Photo Query ---
with tab1:
    st.header("Stage 1 ➡️ Stage 2")
    uploaded_file = st.file_uploader("Choose a photo of a face...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)

        with col2:
            with st.spinner("1. Analyzing emotion..."):
                emotion = model_utils.predict_emotion(emotion_processor, emotion_model, emotion_device, image)
            
            st.success(f"**Detected Emotion:** {emotion.upper()}")

            # This is the DYNAMIC glue!
            dynamic_query = f"What are customers saying who feel {emotion}?"
            st.write(f"**Running dynamic query:** *'{dynamic_query}'*")
            
            with st.spinner(f"2. Searching review database for '{emotion}' reviews..."):
                result = model_utils.query_rag(qa_chain, dynamic_query)
            
            st.subheader("📝 RAG Summary")
            st.write(result['result'])
            
            st.subheader("📚 Retrieved Reviews")
            for doc in result['source_documents']:
                sentiment = model_utils.analyze_sentiment(sentiment_pipe, sentiment_labels, doc.page_content)
                with st.container(border=True):
                    st.markdown(f"**Review (Emotion: {doc.metadata.get('emotion')})**")
                    st.write(f"> {doc.page_content}")
                    st.markdown(f"**Sentiment:** {sentiment}")

# --- TAB 2: Manual Query ---
with tab2:
    st.header("Query the Review Database Manually")
    text_query = st.text_input("Enter your query:", placeholder="e.g., 'What are customers happy about?'")

    if st.button("Search Reviews"):
        if text_query:
            with st.spinner("Searching review database..."):
                result = model_utils.query_rag(qa_chain, text_query)
            
            st.subheader("📝 RAG Summary")
            st.write(result['result'])
            
            st.subheader("📚 Retrieved Reviews")
            for doc in result['source_documents']:
                sentiment = model_utils.analyze_sentiment(sentiment_pipe, sentiment_labels, doc.page_content)
                with st.container(border=True):
                    st.markdown(f"**Review (Emotion: {doc.metadata.get('emotion')})**")
                    st.write(f"> {doc.page_content}")
                    st.markdown(f"**Sentiment:** {sentiment}")
        else:
            st.warning("Please enter a query.")