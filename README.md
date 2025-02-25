# Fake News Detection using Machine Learning

## 📌 Overview
Fake news is a major concern in today's digital era, leading to misinformation and social conflicts. This project aims to detect fake news using machine learning techniques in Python. The dataset contains real and fake news articles, and we use Natural Language Processing (NLP) techniques to preprocess and classify the news as real or fake.

## 🚀 Features
- Data preprocessing and cleaning
- Text vectorization using TF-IDF
- Machine learning models: Logistic Regression & Decision Tree Classifier
- Performance evaluation with accuracy score and confusion matrix
- Data visualization with WordCloud and bar charts

## 📂 Dataset
The dataset consists of:
- `text`: News content
- `class`: 1 (Real News), 0 (Fake News)

📥 **Download Dataset**: [News.csv](https://www.kaggle.com/datasets/subho117/fake-news-detection-using-machine-learning)

## 🛠️ Technologies Used
- **Python**
- **Pandas** (Data manipulation)
- **Seaborn & Matplotlib** (Data visualization)
- **NLTK** (Text preprocessing)
- **Scikit-learn** (Machine learning models & evaluation)

## 📌 Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/Fake-News-Detection-ML.git
   cd fake-news-detection
   ```
2. Run the script:
   ```sh
   python Fake_news_detection.py
   ```

## 🔬 Project Workflow
### 1️⃣ Importing Libraries & Dataset
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

### 2️⃣ Data Preprocessing
- Remove unnecessary columns (`title`, `subject`, `date`)
- Remove stopwords and punctuations
- Shuffle dataset to avoid model bias

### 3️⃣ Data Visualization
- **WordCloud** for real and fake news
- **Bar chart** of most frequent words

### 4️⃣ Feature Extraction
Convert text into numerical vectors using **TF-IDF Vectorizer**:
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorization = TfidfVectorizer()
x_train = vectorization.fit_transform(x_train)
x_test = vectorization.transform(x_test)
```

### 5️⃣ Model Training & Evaluation
#### ✅ Logistic Regression
```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
```
✅ **Accuracy**: 98.93%

#### ✅ Decision Tree Classifier
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
```
✅ **Accuracy**: 99.51%

#### 🔍 Confusion Matrix
```python
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, model.predict(x_test))
metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True]).plot()
```

## 📊 Results
| Model | Training Accuracy | Testing Accuracy |
|--------|----------------|----------------|
| Logistic Regression | 99.37% | 98.93% |
| Decision Tree | 99.99% | 99.51% |


---

