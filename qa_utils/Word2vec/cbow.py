import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import time
import pandas as pd

def run(sentences):
    st.subheader("📘 CBOW Model - Word Embedding Visualization and Similarity Exploration")

    # Preprocessing
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    flat_tokens = [word for sentence in tokenized_sentences for word in sentence]

    if not flat_tokens:
        st.error("❌ No valid words found in your input. Please enter meaningful sentences.")
        return

    # Train Word2Vec CBOW model (sg=0)
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0)

    # Get word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

    if word_vectors.shape[0] < 3:
        st.error("❌ Not enough words for visualization.")
        return

    # Section 1: Select sentences for connection lines
    st.markdown("### ✏️ Select Sentences to Show Connections")
    sentence_labels = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)]
    selected_sentences = st.multiselect(
        "Select sentences to draw connection lines:",
        options=sentence_labels,
        default=sentence_labels[:2]
    )
    selected_indices = [i for i, label in enumerate(sentence_labels) if label in selected_sentences]

    # Section 2: Choose between 2D and 3D plots
    plot_option = st.radio("📈 Choose plot type:", ["2D Plot", "3D Plot"], horizontal=True)

    # Assign colors for each sentence
    cmap = plt.get_cmap('tab20', len(tokenized_sentences))
    hex_colors = [
        '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
        for r, g, b, a in [cmap(i) for i in range(len(tokenized_sentences))]
    ]

    word_colors = []
    for word in model.wv.index_to_key:
        for i, sentence in enumerate(tokenized_sentences):
            if word in sentence:
                word_colors.append(hex_colors[i])
                break

    # PCA Dimension Reduction
    n_components = 2 if plot_option == "2D Plot" else 3
    reduced_vectors = PCA(n_components=n_components).fit_transform(word_vectors)

    # Section 3: Create scatter plot
    if plot_option == "2D Plot":
        scatter = go.Scatter(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            mode='markers+text',
            text=model.wv.index_to_key,
            textposition='top center',
            marker=dict(color=word_colors, size=8),
            hovertemplate="Word: %{text}",
            name="Words"
        )
        fig = go.Figure(data=[scatter])

        for i in selected_indices:
            line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in tokenized_sentences[i] if word in model.wv]
            if len(line_vectors) > 1:
                fig.add_trace(go.Scatter(
                    x=[v[0] for v in line_vectors],
                    y=[v[1] for v in line_vectors],
                    mode='lines',
                    line=dict(color=hex_colors[i], width=1),
                    name=f"Sentence {i+1}"
                ))

        fig.update_layout(
            xaxis_title="X",
            yaxis_title="Y",
            title="2D Visualization of CBOW Word Embeddings",
            width=1000, height=800
        )

    else:  # 3D plot
        scatter = go.Scatter3d(
            x=reduced_vectors[:, 0],
            y=reduced_vectors[:, 1],
            z=reduced_vectors[:, 2],
            mode='markers+text',
            text=model.wv.index_to_key,
            marker=dict(color=word_colors, size=3),
            hovertemplate="Word: %{text}",
            name="Words"
        )
        fig = go.Figure(data=[scatter])

        for i in selected_indices:
            line_vectors = [reduced_vectors[model.wv.key_to_index[word]] for word in tokenized_sentences[i] if word in model.wv]
            if len(line_vectors) > 1:
                fig.add_trace(go.Scatter3d(
                    x=[v[0] for v in line_vectors],
                    y=[v[1] for v in line_vectors],
                    z=[v[2] for v in line_vectors],
                    mode='lines',
                    line=dict(color=hex_colors[i], width=2),
                    name=f"Sentence {i+1}"
                ))

        fig.update_layout(
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            title="3D Visualization of CBOW Word Embeddings",
            width=1000, height=800
        )

    st.plotly_chart(fig, use_container_width=True, key=f"cbow_plot_{time.time()}")

    # Section 4: Explore most similar words
    st.markdown("### 🔍 Explore Similar Words")
    selected_word = st.selectbox("Choose a word to find similar words:", model.wv.index_to_key)

    if selected_word:
        similar_words = model.wv.most_similar(selected_word, topn=10)

        words = [w for w, _ in similar_words]
        similarities = [s for _, s in similar_words]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(words, similarities, color='skyblue')
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Most Similar Words to '{selected_word}'")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Show text list
        st.markdown("#### 📋 Similar Words List")
        for i, (word, score) in enumerate(similar_words, 1):
            st.markdown(f"{i}. **{word}** — Similarity: `{score:.3f}`")

    # Section 5: CBOW vs Skip-gram similarity comparison
    st.markdown("### 📊 Compare Similarity between CBOW and Skip-gram")

    with st.expander("Show CBOW and Skip-gram similarity comparison", expanded=False):
        try:
            # Train Skip-gram model (sg=1)
            skipgram_model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

            if selected_word in skipgram_model.wv and selected_word in model.wv:
                cbow_similar = model.wv.most_similar(selected_word, topn=10)
                skip_similar = skipgram_model.wv.most_similar(selected_word, topn=10)

                cbow_words, cbow_scores = zip(*cbow_similar)
                skip_words, skip_scores = zip(*skip_similar)

                comparison_df = pd.DataFrame({
                    'CBOW_Word': cbow_words,
                    'CBOW_Similarity': cbow_scores,
                    'Skipgram_Word': skip_words,
                    'Skipgram_Similarity': skip_scores
                })

                st.dataframe(comparison_df, use_container_width=True)

                fig, ax = plt.subplots(figsize=(10, 5))
                x = np.arange(len(cbow_words))

                ax.bar(x - 0.2, cbow_scores, width=0.4, label='CBOW', color='lightblue')
                ax.bar(x + 0.2, skip_scores, width=0.4, label='Skip-gram', color='orange')
                ax.set_xticks(x)
                ax.set_xticklabels(cbow_words, rotation=45)
                ax.set_ylabel('Cosine Similarity')
                ax.set_title(f"Similarity Comparison for '{selected_word}'")
                ax.legend()

                st.pyplot(fig)

            else:
                st.warning("⚠️ Word not found in both models. Cannot compare similarity.")

        except Exception as e:
            st.error(f"Error comparing CBOW and Skip-gram: {e}")
