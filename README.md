# 📊 Sentiment Analysis of Public Perception on President Prabowo Subianto’s Leadership

This project implements **sentiment analysis** using **Deep Learning (IndoBERT)** to study Indonesian public perception of President **Prabowo Subianto’s early policies** based on comments from **X (formerly Twitter)** and **YouTube**.

## 📌 Project Overview
We developed an automated NLP pipeline to classify:
- **Sentiment** → Positive, Negative, Neutral  
- **Topic** → Four policies during the first 100 days:
  - Free Nutritious Meal Program
  - 12% VAT (Value Added Tax)
  - Budget Efficiency
  - Danantara

## 🔧 Tech Stack
- **Language**: Python (Google Colab)  
- **Libraries**:  
  - HuggingFace Transformers (IndoBERT)  
  - Scikit-learn  
  - Pandas, Numpy  
  - Matplotlib / Seaborn  
- **Techniques**:  
  - Crawling & Scraping (X API, YouTube API/Apify)  
  - Preprocessing (cleaning, normalization, tokenization, stemming, stopword removal)  
  - Lexicon-based Labeling  
  - SMOTE (Synthetic Minority Oversampling)  

## 🚀 Pipeline
1. **Data Collection** → Crawling and scraping comments from X & YouTube  
2. **Preprocessing** → Cleaning, normalization, tokenization, stemming, stopword removal  
3. **Labeling** → Sentiment lexicon-based classification  
4. **Splitting** → Train (70%), Validation (20%), Test (10%)  
5. **Balancing** → Applied SMOTE to handle class imbalance  
6. **Modeling** → Fine-tuned IndoBERT with multitask learning (sentiment + topic)  
7. **Evaluation** → Accuracy, Precision, Recall, F1-Score, Confusion Matrix  

## 📊 Results
- **Topic Classification Accuracy**: **81.38%**  
- **Sentiment Classification Accuracy**: **73.95%**  
- **Overall Public Sentiment**:  
  - Negative → 56.3%  
  - Positive & Neutral → 43.7%  
- Negative sentiment dominated especially on:  
  - Budget Efficiency (63.4%)  
  - VAT 12% (58.3%)  

## 📂 Project Structure
