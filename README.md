# Fake News Detection Web Application

An **AI-powered Fake News Detection Web App** that analyzes news articles and predicts whether they are **Fake or Real** using **Natural Language Processing (NLP)** and **Machine Learning**.

This project combines **Machine Learning, Flask API, and React.js** to build a full-stack AI application.

---

# Features

*  Detect whether a news article is **Fake or Real**
*  Displays **prediction confidence score**
*  Real-time prediction using **Flask API**
*  Modern **React UI with premium design**
*  Machine Learning model trained on real-world dataset

---

#  Tech Stack

### Machine Learning

* Python
* Scikit-learn
* TF-IDF Vectorizer
* Logistic Regression
* Pandas

### Backend

* Flask
* Flask-CORS
* Joblib

### Frontend

* React.js
* Axios
* CSS (Modern UI Design)

---

#  Model Performance

The model was trained using the **Fake and Real News Dataset from Kaggle**.

**Training Accuracy**

```
Accuracy: 0.9846325167037862
```

✔ Achieved **98.46% accuracy** using **TF-IDF Vectorization + Logistic Regression**

---

#  Project Structure

```
fake-news-detector
│
├── backend
│   ├── app.py
│   ├── trainmodel.py
│   ├── model.pkl
│   ├── vectorizer.pkl
│   └── requirements.txt
│
├── dataset
│   ├── Fake.csv
│   └── True.csv
│
├── frontend
│   ├── public
│   └── src
│       ├── App.js
│       ├── App.css
│       └── index.js
│
└── README.md
```

---

#  Installation & Setup

##  Clone the Repository

```
git clone https://github.com/Dhanush-1213/fake-news-detector-react-flask.git
cd fake-news-detector-react-flask
```

---

#  Backend Setup

```
cd backend
pip install -r requirements.txt
python app.py
```

Backend runs on:

```
http://127.0.0.1:5000
```

---

#  Frontend Setup

```
cd frontend
npm install
npm start
```

Frontend runs on:

```
http://localhost:3000
```

---

#  How It Works

```
User enters news article
        ↓
React Frontend
        ↓
Flask REST API
        ↓
TF-IDF Vectorization
        ↓
Logistic Regression Model
        ↓
Prediction (Fake / Real) + Confidence Score
```

---

#  Example

### Input

```
Breaking news: Scientists discovered water on Mars
```

### Output

```
REAL
Confidence: 92%
```

---

#  Future Improvements

* Explainable AI using **LIME / SHAP**
* News URL analyzer
* Deep Learning model using **BERT**
* Online deployment (Vercel + Render)
* Fake news keyword highlighting

---

#  Author

**Dhanush K**

Machine Learning & AI Enthusiast
