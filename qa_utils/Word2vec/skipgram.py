import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.decomposition import PCA
import plotly.graph_objs as go
import time

def run(sentences):
    st.subheader("⚙️ Skip-gram Model - Word Embedding Visualization and Similarity Exploration")

    # Preprocessing
    tokenized_sentences = [simple_preprocess(sentence) for sentence in sentences]
    flat_tokens = [word for sentence in tokenized_sentences for word in sentence]

    if not flat_tokens:
        st.error("❌ No valid words found in your input. Please enter meaningful sentences.")
        return

    # Train Skip-gram model (sg=1)
    model = Word2Vec(tokenized_sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

    # Word vectors
    word_vectors = np.array([model.wv[word] for word in model.wv.index_to_key])

    if word_vectors.shape[0] < 3:
        st.error("❌ Not enough words for visualization.")
        return

    # --- Section 1: Sentence selection for lines ---
    st.markdown("### ✏️ Select Sentences to Show Connections")
    sentence_labels = [f"Sentence {i+1}: {s}" for i, s in enumerate(sentences)]
    selected_sentences = st.multiselect(
        "Select sentences to draw connection lines:",
        options=sentence_labels,
        default=sentence_labels[:2]
    )

    selected_indices = [i for i, label in enumerate(sentence_labels) if label in selected_sentences]

    # --- Section 2: Choose 2D or 3D ---
    plot_option = st.radio("📈 Choose plot type:", ["2D Plot", "3D Plot"], horizontal=True)

    # Color for each sentence
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

    # Dimension reduction
    n_components = 2 if plot_option == "2D Plot" else 3
    reduced_vectors = PCA(n_components=n_components).fit_transform(word_vectors)

    # --- Section 3: Scatter plot ---
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
                    showlegend=True,
                    name=f"Sentence {i+1}"
                ))

        fig.update_layout(
            xaxis_title="X",
            yaxis_title="Y",
            title="2D Visualization of Skip-gram Word Embeddings",
            width=1000, height=800
        )

    else:  # 3D Plot
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
            title="3D Visualization of Skip-gram Word Embeddings",
            width=1000, height=800
        )

    st.plotly_chart(fig, use_container_width=True, key=f"skipgram_plot_{time.time()}")

    # --- Section 4: Explore similar words ---
    st.markdown("### 🔍 Explore Similar Words")
    selected_word = st.selectbox("Choose a word to find similar words:", model.wv.index_to_key)

    if selected_word:
        similar_words = model.wv.most_similar(selected_word, topn=10)

        words = [w for w, _ in similar_words]
        similarities = [s for _, s in similar_words]

        # ➡️ 畫 bar chart
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(words, similarities, color='skyblue')
        ax.set_ylabel("Cosine Similarity")
        ax.set_title(f"Most Similar Words to '{selected_word}'")
        ax.set_ylim(0, 1)
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # ➡️ 顯示文字版 Top 10 相似詞
        st.markdown("#### 📋 Similar Words List")
        for i, (word, score) in enumerate(similar_words, 1):
            st.markdown(f"{i}. **{word}** — Similarity: `{score:.3f}`")

