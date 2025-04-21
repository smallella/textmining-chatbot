import fitz  # PyMuPDF
import pymupdf
import streamlit as st
import re
import jieba
from collections import Counter
from esg_keywords_zh import E_KEYWORDS_ZH, S_KEYWORDS_ZH, G_KEYWORDS_ZH
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

def load_pdf(pdf_path):
    """Loads a PDF file using PyMuPDF.

    Args:
        pdf_path: The path to the PDF file.

    Returns:
        A PyMuPDF document object if successful, None otherwise.
    """
    try:
        doc = pymupdf.open(pdf_path, filetype="pdf")
        return doc
    except FileNotFoundError:
        print(f"Error: PDF file not found at '{pdf_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading the PDF: {e}")
        return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'-\s+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    return text.strip()

def extract_text_by_page(doc, max_pages=40, skip_pages=[]):
    formatted_full_text = []
    total_items = len(doc)
    total_pages = min(len(doc), max_pages)

    for page_number, page in enumerate(doc):
        if page_number >= max_pages:
            break
        if int(page_number) + 1 in skip_pages:
            print(f"Skip page {page_number+1}")
            continue

        try:
            this_text = clean_text(page.get_text())

            # Extract tables
            tables = page.find_tables()
            for table in tables:
                df = table.to_pandas()
                this_text += "\nTable:\n" + df.to_string() + "\n"
            print(f"Text length in page {page_number+1}: {len(this_text)}")

            formatted_full_text.append({
                "page": page_number + 1,
                "content": this_text
            })

            # Update progress
            progress = (page_number + 1) / total_pages
            print(f"Progress: {round(progress*100)}%")
            print(f"Processing {page_number + 1}/{total_pages} pages with document (max_pages:{max_pages})...")

        except Exception as e:
            print(f"(extract_text_by_page) Error processing page {page}: {e}")

    print("Processing complete!")

    return formatted_full_text

def pdf_upload_section():
    uploaded_file = st.file_uploader("📄 Upload a PDF file", type=["pdf"])

    # 若已解析 pdf 就不要重複執行
    if uploaded_file and "pdf_text" not in st.session_state:
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        extracted = extract_text_by_page(doc, max_pages=len(doc))
        st.session_state["pdf_text"] = extracted
        st.success("✅ PDF uploaded and parsed successfully!")

    # Clear button
    if "pdf_text" in st.session_state:
        if st.button("🗑️ Clear PDF"):
            del st.session_state["pdf_text"]
            st.rerun()

def get_pdf_context(page="all"):
    if page != "all" and "pdf_text" in st.session_state:
        # Return specific page content
        for p in st.session_state["pdf_text"]:
            if p["page"] == page:
                return f"[Page {p['page']}]: {p['content']}"
        return f"Page {page} not found in the PDF."
    elif page == "all" and "pdf_text" in st.session_state:
        # Return part of the text for context
        # return "\n\n".join([f"[Page {p['page']}]: {p['content'][:300]}..." for p in st.session_state["pdf_text"]])
        # Return whole text
        return "\n\n".join([f"[Page {p['page']}]: {p['content']}" for p in st.session_state["pdf_text"]])
    else:
        return ""

def generate_response(prompt):
    pdf_context = get_pdf_context()
    original_prompt = prompt
    prompt = prompt.strip().lower()

    if prompt not in ["show content", "clustering analysis", "esg analysis","show pdf page"] and "which dimension is emphasized" not in prompt:
        return (
            "📝 It looks like your prompt might not match the expected operations.\n\n"
            "💡 Try entering prompts like:\n"
            "- Show content\n"
            "- Show pdf page <num>\n"
            "- Clustering analysis\n"
            "- ESG analysis\n\n"
            "- Which dimension is emphasized\n\n"
            "📄 Also, make sure you've uploaded a PDF file first!"
        )

    if not pdf_context:
        return f"Please upload a PDF file to get context."
    elif prompt == "show content":
        # {pdf_context[:1000]}
        return f"""
        🤖 Here's what I found from the uploaded PDF:\n
        {pdf_context}
        ----------------------------------\n
        """
    elif "show pdf page" in prompt:
        match = re.search(r"show pdf page (\d+)", prompt)
        if match:
            page_number = int(match.group(1))
            return get_pdf_context(page=page_number)
        else:
            return "⚠️ Please specify the page number, e.g., `Show PDF page 2`."
    elif prompt == "clustering analysis":

        # "colab code"

        return f"📊 Working on clustering analysis..."
    elif prompt == "esg analysis":

        # "colab code"

        return f"🌱 Working on ESG analysis..."
    elif prompt == "which dimension is emphasized":
        # 整份報告所有文字合併
        all_text = " ".join([p["content"] for p in st.session_state["pdf_text"]])

        # 中文斷詞，移除空白、標點、數字，且僅保留長度 ≥ 2 的詞
        words = [w.strip() for w in jieba.lcut(all_text) if re.match(r"^[\u4e00-\u9fff]{2,}$", w)]
        freq = Counter(words)

        # ESG 各面向關鍵詞字頻
        e_words = {w: freq[w] for w in E_KEYWORDS_ZH if w in freq}
        s_words = {w: freq[w] for w in S_KEYWORDS_ZH if w in freq}
        g_words = {w: freq[w] for w in G_KEYWORDS_ZH if w in freq}

        e_count = sum(e_words.values())
        s_count = sum(s_words.values())
        g_count = sum(g_words.values())
        total = e_count + s_count + g_count + 1e-5  # 防止除以 0

        e_ratio = e_count / total
        s_ratio = s_count / total
        g_ratio = g_count / total

        max_dim = max([
            ("Environmental", e_ratio),
            ("Social", s_ratio),
            ("Governance", g_ratio)
        ], key=lambda x: x[1])

        def format_top_keywords(keywords_dict, top_n=10):
            top_items = sorted(keywords_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
            return "\n".join([f"- {k}: {v}" for k, v in top_items]) if top_items else "No keywords found."

        # ⬇ ESG 總結說明
        st.markdown(f"""📊 ESG Keyword Frequency (Chinese Text Analysis)  
Environmental Ratio: {e_ratio:.1%}  
Social Ratio: {s_ratio:.1%}  
Governance Ratio: {g_ratio:.1%}  

🔍 This report places the greatest emphasis on: {max_dim[0]}
---  
""")
        # 比例長條圖
        ratio_df = pd.DataFrame({
            'Dimension': ['Environmental', 'Social', 'Governance'],
            'Ratio': [e_ratio, s_ratio, g_ratio]
        })
        # 繪圖
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(ratio_df['Dimension'], ratio_df['Ratio'], color='skyblue')

        # 加上數值標註
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f'{yval:.1%}', ha='center', va='bottom', fontsize=10)

        # 調整 X 軸文字方向為水平
        plt.xticks(rotation=0)
        ax.set_ylim(0, 1)  # Y 軸最大值設為 1 方便展示百分比
        st.pyplot(fig)

        # 字雲顯示（可選：只用 ESG 的關鍵詞，或整份報告常見詞）
        st.markdown("☁️ ESG Keyword Word Cloud")
        wordcloud = WordCloud(font_path='msjh.ttc',  # 需要支援中文字體，如微軟正黑體
                            width=800, height=400,
                            background_color='white').generate_from_frequencies(freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(plt)

        # ⬇ 三欄並排顯示關鍵詞前十名
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("🟢 Top Environmental Keywords")
            st.markdown(format_top_keywords(e_words, top_n=10))

        with col2:
            st.markdown("🟠 Top Social Keywords")
            st.markdown(format_top_keywords(s_words, top_n=10))

        with col3:
            st.markdown("🔵 Top Governance Keywords")
            st.markdown(format_top_keywords(g_words, top_n=10))

        return ""


    # 加一個 fallback return，防止漏掉時回傳 None
    return f"⚠️ Unexpected issue of prompt - ```{original_prompt}```. Please try again."