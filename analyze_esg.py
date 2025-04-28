import jieba
import re
import json
from collections import Counter

def load_esg_keywords(json_path="esg_keywords.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        keywords = json.load(f)
    return keywords

def analyze_esg_text(text: str, esg_keywords: dict, top_n: int = 10):
    words = [w.strip() for w in jieba.lcut(text) if re.match(r"^[\u4e00-\u9fff]{2,}$", w)]
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
    import pandas as pd

    esg_keywords = load_esg_keywords(json_path)
    results = analyze_esg_text(text, esg_keywords, top_n=top_n)

    st.markdown(f"# 📄 ESG Analysis for: `{filename}`")
    st.markdown(f"""📊 ESG Keyword Frequency (Chinese Text Analysis)  
Environmental Ratio: {results['Environmental']['ratio']:.1%}  
Social Ratio: {results['Social']['ratio']:.1%}  
Governance Ratio: {results['Governance']['ratio']:.1%}  

🔍 This report places the greatest emphasis on: {max(results.items(), key=lambda x: x[1]['ratio'])[0]}
---
""")

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

    st.markdown("☁️ ESG Keyword Word Cloud")
    # full_freq = Counter()
    # for dim in results:
    #     full_freq.update(results[dim]["keywords"])
    words = [w.strip() for w in jieba.lcut(text) if re.match(r"^[\u4e00-\u9fff]{2,}$", w)]
    full_freq = Counter(words)
    wordcloud = WordCloud(font_path="msjh.ttc", background_color="white", width=800, height=400)\
        .generate_from_frequencies(full_freq)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    st.pyplot(plt)


    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("🟢 Top Environmental Keywords")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Environmental"]["keywords"].items()]))
    with col2:
        st.markdown("🟠 Top Social Keywords")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Social"]["keywords"].items()]))
    with col3:
        st.markdown("🔵 Top Governance Keywords")
        st.markdown("\n".join([f"- {k}: {v}" for k, v in results["Governance"]["keywords"].items()]))

    return results
