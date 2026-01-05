# 🧾 TaxBot Assistant

A Gemini-powered chatbot that helps users understand Indian tax regimes (old vs new) using real PDF documents.

## 📦 Files
- `app.py`: Main Flask backend
- `build_vector_db.py`: Embeds the PDF
- `templates/index.html`: Chat UI
- `tax project document.pdf`: Source for tax rules
- `.env`: Contains `GOOGLE_API_KEY` (NOT uploaded)

## ⚙️ Setup
```bash
pip install -r requirements.txt
python build_vector_db.py
python app.py
