import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import time

def run(sentences):
    st.subheader("🌱 Negative Sampling - Explore Word Relationships from ESG Report")

    # Preprocess the input sentences
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    flat_tokens = [word for sentence in tokenized_sentences for word in sentence]

    if not flat_tokens:
        st.error("❌ No valid words found. Please input valid ESG-related sentences.")
        return

    # Train a Word2Vec model using Skip-gram + Negative Sampling
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1, negative=10)

    st.markdown("### 🎯 Select a Word to Explore")
    selected_word = st.selectbox(
        "Choose a word from the vocabulary:",
        options=model.wv.index_to_key
    )

    if selected_word:
        similar_words = model.wv.most_similar(selected_word, topn=10)

        words = [w for w, _ in similar_words]
        similarities = [s for _, s in similar_words]

        # Plot Top 10 Similar Words
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(words, similarities, color='seagreen')
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Top 10 Words Similar to '{selected_word}'")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

        # Show Similar Words in text form
        st.markdown("### 📋 Similar Words List")
        for i, (word, score) in enumerate(similar_words, 1):
            st.markdown(f"{i}. **{word}** — Similarity: `{score:.3f}`")
