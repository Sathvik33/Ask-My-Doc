# 📄 PDF Bot – Donut DocVQA

--- ***This is PDF Bot*** ---

A simple Streamlit app that uses **Donut (Document Understanding Transformer)** for **document question answering**.  
No OCR needed — Donut reads the document directly and answers questions.

---

## 🚀 Features
- 📂 Upload any PDF
- 💬 Ask natural language questions about your document
- 📑 Get answers **per page**
- ⚡ Works on both CPU & GPU
- 🔍 OCR-free, powered by **Donut DocVQA**

---

## 📦 Installation

pip install -r requirements.txt

Run the app

streamlit run app.py

---

Requirements

The app needs:

Python 3.8+

PyMuPDF
 for PDF reading

Transformers
 for model loading

Streamlit
 for the web interface


 ---

 Model Used

We use naver-clova-ix/donut-base-finetuned-docvqa from HuggingFace,
a transformer model designed for document image understanding & question answering.

Model link: https://huggingface.co/naver-clova-ix/donut-base-finetuned-docvqa

---

How It Works

Upload a PDF file

The PDF is converted to page images

Donut processes each page

You get answers for your question per page
---

