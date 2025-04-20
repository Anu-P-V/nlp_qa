import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from simpletransformers.question_answering import QuestionAnsweringModel
from datasets import load_dataset
import logging

# Disable logging warnings
logging.basicConfig(level=logging.WARNING)

# Load (and cache) the BERT QA model
@st.cache_resource
def load_model():
    return QuestionAnsweringModel(
        model_type="bert",
        model_name="bert-base-uncased",
        use_cuda=False,
        args={
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
        }
    )

model = load_model()

# Load (and cache) the first 500 contexts from SQuAD v1.1
@st.cache_resource
def load_squad_data():
    dataset = load_dataset("squad")
    return dataset["train"].select(range(500))["context"]

knowledge_base = load_squad_data()

# Streamlit UI
st.title("📚 Smart QA — Powered by SQuAD + BERT")

question = st.text_input("Ask your question (no manual context needed)")

if question:
    # 1) Find the most relevant context via TF‑IDF + cosine similarity
    tfidf = TfidfVectorizer()
    docs = knowledge_base + [question]
    vectors = tfidf.fit_transform(docs)
    scores = cosine_similarity(vectors[-1], vectors[:-1])
    best_context = knowledge_base[scores.argmax()]

    st.write("📌 **Best Context Found:**")
    st.info(best_context)

    # 2) Run the model to extract the answer from that context
    input_data = [{
        "context": best_context,
        "qas": [{"question": question, "id": "1"}]
    }]
    result = model.predict(input_data)
    answer = result[0]["answer"][0]

    st.success(f"💬 **Answer:** {answer}")
