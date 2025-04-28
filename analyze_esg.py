import jieba
import re
import json
from collections import Counter

# English stopwords (you can expand this list)
ENGLISH_STOPWORDS = set([
    "the", "and", "is", "in", "to", "for", "of", "on", "at", "by", "with",
    "an", "be", "this", "that", "from", "as", "are", "it", "or", "which", "has", "was", "a","none"
])

def load_esg_keywords(json_path="esg_keywords.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    return keywords

def extract_words(text: str):
    """Auto detect whether the text is Chinese or English, and segment it."""
    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_ratio = len(chinese_characters) / (len(text) + 1e-5)

    if chinese_ratio > 0.3:
        # Chinese text
        words = [w.strip() for w in jieba.lcut(text) if re.match(r"^[\u4e00-\u9fff]{2,}$", w)]
    else:
        # English text
        words = [w.strip().lower() for w in text.split() if w.isalpha()]
        words = [w for w in words if w not in ENGLISH_STOPWORDS]

    return words

def analyze_esg_text(text: str, esg_keywords: dict, top_n: int = 10):
    words = extract_words(text)
    freq = Counter(words)

    results = {}
    for dim in ["Environmental", "Social", "Governance"]:
        keywords = esg_keywords.get(dim, [])
        filtered = {w: freq[w] for w in keywords if w in freq}
        count = sum(filtered.values())
        results[dim] = {
            "count": count,
            "keywords": dict(sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_n])
        }

    total = sum([results[dim]["count"] for dim in results]) + 1e-5
    for dim in results:
        results[dim]["ratio"] = results[dim]["count"] / total

    return results

def display_esg_analysis(text: str, filename: str, json_path="esg_keywords.json", top_n=10):
    import streamlit as st
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud
    import os

    esg_keywords = load_esg_keywords(json_path)
    results = analyze_esg_text(text, esg_keywords, top_n=top_n)

    # Safer Markdown without emoji to avoid UnicodeEncodeError
    st.markdown(f"# ESG Analysis for `{filename}`")
    st.markdown(f"""📊 **ESG Keyword Frequency (Text Analysis)**  
- Environmental Ratio: {results['Environmental']['ratio']:.1%}  
- Social Ratio: {results['Social']['ratio']:.1%}  
- Governance Ratio: {results['Governance']['ratio']:.1%}  

🔍 **Main Focus Area:** {max(results.items(), key=lambda x: x[1]['ratio'])[0]}
---
""")

    # ESG Ratio bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        ["Environmental", "Social", "Governance"],
        [results["Environmental"]["ratio"], results["Social"]["ratio"], results["Governance"]["ratio"]],
        color='skyblue'
    )
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.1%}', ha='center')
    plt.xticks(rotation=0)
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    # Word Cloud Section
    st.markdown("## ☁️ ESG Keyword Word Cloud")
    words = extract_words(text)
    full_freq = Counter(words)

    chinese_characters = re.findall(r'[\u4e00-\u9fff]', text)
    chinese_ratio = len(chinese_characters) / (len(text) + 1e-5)

    if chinese_ratio > 0.3:
        # Chinese wordcloud
        font_path = "fonts/NotoSansTC-VariableFont_wght.ttf"
        if not os.path.exists(font_path):
            font_path = None  # fallback
    else:
        font_path = None

    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        width=800,
        height=400
    ).generate_from_frequencies(full_freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)

    # ESG Top Keywords Columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 **Top Environmental Keywords**")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Environmental"]["keywords"].items()]))
    with col2:
        st.markdown("🟠 **Top Social Keywords**")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Social"]["keywords"].items()]))
    with col3:
        st.markdown("🔵 **Top Governance Keywords**")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Governance"]["keywords"].items()]))

    return results
