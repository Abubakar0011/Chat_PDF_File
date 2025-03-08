# 📄 Chat with Multiple PDFs using Gemini 💡

## 🚀 Overview
Turn your PDFs into an interactive knowledge base! This app lets you 
chat with multiple PDFs using Google's Gemini AI. Just upload files, 
ask questions, and get instant AI-powered answers.

## ⚡ Features
✅ Upload & process multiple PDFs
✅ AI-powered Q&A using Gemini 1.5 Pro
✅ Smart text extraction & chunking
✅ Fast, efficient search with FAISS
✅ Simple & interactive Streamlit UI

## 🔧 Installation
```sh
pip install -r requirements.txt
```

Add your Gemini API key to a `.env` file:
```
GEMINI_API_KEY=your_api_key_here
```

Run the app:
```sh
streamlit run app.py
```

## 💡 How It Works
1️⃣ Upload your PDFs 📂
2️⃣ AI processes & embeds the content 🤖
3️⃣ Ask questions & get instant answers! 💬

## 📌 Tech Stack
- **Streamlit** (UI)
- **PyPDF2** (Text extraction)
- **LangChain** (AI pipeline)
- **FAISS** (Vector search)
- **Google Gemini AI** (LLM)

## 🛠 Troubleshooting
If you see `ModuleNotFoundError: langchain_community`:
```sh
pip install -U langchain-community
```

## 🎯 Future Upgrades
✨ Support for more file types
✨ Enhanced UI & search filters
✨ Chat history & session storage

## 🤝 Contribute
Pull requests & ideas are welcome!

