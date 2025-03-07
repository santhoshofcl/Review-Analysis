# 📢 Automated Review Analysis & Fake Review Detection

## 🚀 Project Overview
Fake reviews are a major problem in online marketplaces, influencing consumer decisions unfairly. This project aims to **detect fake reviews** using **Natural Language Processing (NLP) and Machine Learning (ML)**.  

It provides a **web interface** where users can submit a review, and the system will classify it as **real (0) or fake (1)**.

---

## 📊 Dataset Details (`reviews_dataset.csv`)
The dataset contains **customer reviews** labeled as **fake (1) or genuine (0)**.

| Review | Fake (1) / Genuine (0) |
|--------|-----------------------|
| "This product is amazing! Works as expected." | 0 |
| "Best perfume ever! Ordered 5 times already!" | 1 |
| "Worst experience ever. The phone stopped working in a week." | 0 |
| "Great deal! Great product! Great seller!" | 1 |

👉 **Fake reviews (1)** tend to use **overly positive words, repetitive phrases, and exaggerated claims**.  
👉 **Genuine reviews (0)** provide **balanced feedback** and are more detailed.

---

## ⚙️ Installation & Setup
### **2️⃣ Install Dependencies**
Make sure you have **Python 3.8+** installed. Then run:
```bash
pip install -r requirements.txt
```

### **3️⃣ Run the Web Application**
```bash
python app.py
```
**Now, open `http://127.0.0.1:5000/` in your browser.**  

---

## 🖥️ Web Interface
🔹 **Input**: Enter a review in the text box.  
🔹 **Process**: The model analyzes the review using NLP techniques.  
🔹 **Output**: It classifies the review as **Fake (1) or Genuine (0)** and displays confidence scores.

---

## 🧠 How It Works
1. **User submits a review.**  
2. **Pretrained ML Model** processes the review text.  
3. **NLP Analysis & Feature Extraction** (TF-IDF, Word Embeddings, Stopword Removal, etc.).  
4. **Prediction:**  
   - `0` → Genuine Review ✅  
   - `1` → Fake Review ❌  
5. **Result displayed on the web interface.**

---

## 📌 Technologies Used
✅ **Frontend:** HTML, CSS, JavaScript  
✅ **Backend:** Python (Flask)  
✅ **ML Model:** NLP with TensorFlow/PyTorch, Scikit-Learn  
✅ **Dataset Processing:** Pandas, NumPy  
✅ **Text Processing:** spaCy, NLTK  

---

## 🔍 Machine Learning Model
This project uses **Supervised Learning (Logistic Regression, SVM, or Transformers like BERT)** trained on labeled review datasets.

### 📌 ML Pipeline:
1️⃣ **Data Preprocessing:**  
   - Lowercasing, stopword removal, lemmatization  
   - TF-IDF Vectorization or Word Embeddings (Word2Vec, BERT)  
2️⃣ **Model Training:**  
   - Train a classification model (Logistic Regression, SVM, BERT, etc.)  
3️⃣ **Evaluation:**  
   - Accuracy, Precision-Recall, F1-score  

### **To Train the Model Manually**
If you want to retrain the model, run:
```bash
python model_training.py
```
👉 This will save the model as `fake_review_model.pkl`.

---

## 🔥 Features
✔ **Real-time Fake Review Detection**  
✔ **Interactive Web Interface**  
✔ **Trained NLP Model for Text Analysis**  
✔ **Easy Deployment on Cloud (AWS/GCP)**  
✔ **Can be Integrated with E-commerce Websites**  

---

## 📝 Future Enhancements
🔹 **Improve ML accuracy with Transformers (BERT, GPT, etc.)**  
🔹 **Implement Real-time Review Analysis using APIs**  
🔹 **Add Sentiment Analysis for Deeper Insights**  
🔹 **Deploy on AWS/GCP with a Scalable Backend**  

---

## 🛠️ Contributors
👤 **Santhosh Krishnan**  
🔗 [LinkedIn](https://www.linkedin.com/in/santhosh-krishnan-802574292) | 🌎 [Website](https://santhoshofcl.info)  

---

## 📜 License
This project is **open-source** under the **MIT License**.  

---

## 🌟 Support & Contributions
📌 **Found a bug?** Create an issue on GitHub.  
📌 **Want to contribute?** Fork the repo and send a PR.  
📌 **Need help?** Contact me on [LinkedIn](https://www.linkedin.com/in/santhosh-krishnan-802574292).  

