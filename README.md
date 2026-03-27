
# 🔍 Fake News Detector

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Accuracy](https://img.shields.io/badge/Accuracy-98.46%25-22c55e?style=for-the-badge)](/)

<br/>

> **An AI-powered full-stack web app that classifies any news article as Fake or Real in real time — with a confidence score, powered by TF-IDF + Logistic Regression and a React + Flask architecture.**

<br/>

[🚀 Getting Started](#-getting-started) · [⚙️ How It Works](#️-how-it-works) · [📡 API Reference](#-api-reference) · [🔬 Model Details](#-model-details) · [🔮 Roadmap](#-roadmap)

</div>

---

## 📌 Overview

**Fake News Detector** is a full-stack AI project that lets users paste any news article text and get an instant **FAKE** or **REAL** prediction along with a **confidence score**.

Built with a clean separation of concerns:
- **ML layer** — Logistic Regression trained on ~44,000 real-world articles
- **Backend** — Flask REST API serving predictions via `/predict`
- **Frontend** — React.js UI with Axios for real-time communication

Trained on the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset), achieving **98.46% test accuracy**.

---

## ✨ Features

- 🧠 **AI Classification** — Detects Fake vs Real news using a production-ready ML pipeline
- 📊 **Confidence Score** — Every prediction comes with a probability (e.g., 97% confident)
- ⚡ **Real-Time** — Flask API responds instantly; no page reloads
- 🎨 **Modern React UI** — Premium, responsive frontend with clean design
- 📁 **Bulk-ready Backend** — API accepts any text input, easy to extend for batch use
- 🔒 **CORS Enabled** — Frontend and backend communicate seamlessly across origins

---

## 🏗️ Project Structure

```
fake-news-detector/
│
├── backend/
│   ├── app.py              # Flask REST API — exposes GET / and POST /predict
│   ├── trainmodel.py       # Full ML training script (TF-IDF + Logistic Regression)
│   ├── model.pkl           # Saved trained Logistic Regression model
│   ├── vectorizer.pkl      # Saved fitted TF-IDF vectorizer
│   └── requirements.txt    # Python dependencies
│
├── dataset/
│   ├── Fake.csv            # ~23,000 fake news articles (Kaggle)
│   └── True.csv            # ~21,000 real news articles (Kaggle)
│
├── frontend/
│   ├── public/
│   └── src/
│       ├── App.js          # Main React component — form input + result display
│       ├── App.css         # UI styling
│       └── index.js        # React entry point
│
└── README.md
```

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│   User pastes news article text in React UI             │
│                         ↓                               │
│   Axios sends POST /predict to Flask (port 5000)        │
│                         ↓                               │
│   Flask loads model.pkl + vectorizer.pkl                │
│                         ↓                               │
│   TF-IDF transforms raw text → numeric feature vector   │
│   (stop_words="english", max_df=0.7)                    │
│                         ↓                               │
│   Logistic Regression predicts: "FAKE" or "REAL"        │
│   predict_proba() returns confidence score              │
│                         ↓                               │
│   JSON response → React renders result + confidence     │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.8+ |
| Node.js | 16+ |
| npm | 8+ |

---

### 1. Clone the Repository

```bash
git clone https://github.com/Dhanush-1213/fake-news-detector-react-flask.git
cd fake-news-detector-react-flask
```

---

### 2. Backend Setup

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** Pre-trained `model.pkl` and `vectorizer.pkl` are already included. Skip straight to running the server unless you want to retrain.

#### (Optional) Retrain the Model from Scratch

```bash
python trainmodel.py
```

This will:
- Load `Fake.csv` and `True.csv` from `../dataset/`
- Label fake articles as `"FAKE"`, real as `"REAL"`
- Vectorize text using TF-IDF
- Train Logistic Regression with `max_iter=1000`
- Print accuracy on the test set
- Save `model.pkl` and `vectorizer.pkl`

#### Start the Flask Server

```bash
python app.py
```

✅ Backend live at: `http://127.0.0.1:5000`

---

### 3. Frontend Setup

Open a new terminal:

```bash
cd frontend
npm install
npm start
```

✅ Frontend live at: `http://localhost:3000`

---

## 📡 API Reference

### `POST /predict`

Accepts a news article and returns a prediction with confidence.

**Request**

```http
POST http://127.0.0.1:5000/predict
Content-Type: application/json
```

```json
{
  "text": "Breaking news: Scientists discovered water on Mars."
}
```

**Response**

```json
{
  "prediction": "REAL",
  "probability": 0.9241
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | `string` | `"FAKE"` or `"REAL"` |
| `probability` | `float` | Model confidence — value between `0.0` and `1.0` |

---

### `GET /`

Health check endpoint.

**Response**

```
Fake News Detection API Running
```

---

## 🔬 Model Details

### Dataset

| File | Label | Size |
|------|-------|------|
| `Fake.csv` | `FAKE` | ~23,000 articles |
| `True.csv` | `REAL` | ~21,000 articles |
| **Combined** | — | **~44,000 articles** |

Source: [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

### Training Pipeline

```python
# 1. Load & label
fake["label"] = "FAKE"
real["label"]  = "REAL"
data = pd.concat([fake, real])

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"],
    test_size=0.2, random_state=42
)

# 3. Vectorize
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluate
print("Accuracy:", accuracy_score(y_test, model.predict(X_test_vec)))
# → Accuracy: 0.9846325167037862
```

---

### Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.46%** |
| Algorithm | Logistic Regression |
| Vectorizer | TF-IDF (`stop_words="english"`, `max_df=0.7`) |
| Train/Test Split | 80% / 20% (`random_state=42`) |
| Training data | ~35,200 articles |
| Test data | ~8,800 articles |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| ML Model | Scikit-learn — Logistic Regression |
| Vectorizer | TF-IDF (Scikit-learn) |
| Backend | Flask + Flask-CORS |
| Model Storage | Joblib (`.pkl` files) |
| Frontend | React.js |
| HTTP Client | Axios |
| Styling | CSS |

---

## 💡 Example

**Input**
```
Scientists at NASA confirmed the discovery of liquid water beneath the 
surface of Mars, marking the biggest breakthrough in planetary science 
in decades.
```

**Output**
```json
{
  "prediction": "REAL",
  "probability": 0.92
}
```

---

## 🔮 Roadmap

- [ ] **Explainability** — Highlight words that most influenced the prediction (LIME / SHAP)
- [ ] **URL Analyzer** — Paste a news link instead of raw text
- [ ] **BERT Upgrade** — Replace Logistic Regression with a transformer model for higher accuracy
- [ ] **Keyword Highlighting** — Visually mark suspicious phrases in the article
- [ ] **Deployment** — Host on Vercel (frontend) + Render (backend)
- [ ] **Browser Extension** — Analyze articles directly while browsing the web
- [ ] **Multi-language Support** — Extend beyond English-language articles

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "feat: add your feature"`
4. Push to your fork: `git push origin feature/your-feature`
5. Open a Pull Request

---


## 👨‍💻 Author

**Dhanush K** — Machine Learning & Full-Stack AI Enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-Dhanush--1213-181717?style=flat&logo=github)](https://github.com/Dhanush-1213)

---

<div align="center">
  <sub>⭐ If this project helped you, please star the repository!</sub>
</div>
