# 🛍️ NLP-Based Clothing Recommendation System

This project showcases a complete pipeline for building a Natural Language Processing (NLP)-powered recommendation system using real-world customer reviews from an online clothing retailer. The system is capable of predicting whether a product will be recommended by a customer based on their review, using multiple text classification approaches.

## 🔍 Project Overview

E-commerce platforms increasingly rely on product reviews and recommendation systems to personalize customer experiences and improve sales. This project leverages natural language processing to pre-process customer reviews, extract meaningful insights, and classify whether a product is likely to be recommended.

The project is divided into three main components:

1. **Text Pre-processing**
2. **Feature Representation**
3. **Machine Learning-based Classification**

---

## 📁 Dataset

The dataset used contains approximately 19,600 women’s clothing reviews, including the following key fields:

* `Title`: A short title of the review
* `Review Text`: The full customer review
* `Recommended`: Binary label (`1` = recommended, `0` = not recommended)

*Original Source:* [Kaggle Women's E-Commerce Clothing Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews) (with modifications)

---

## 🧹 1. Text Pre-processing

The first stage of the project focuses on cleaning and preparing the raw review text:

* Tokenization using regex `r"[a-zA-Z]+(?:[-'][a-zA-Z]+)?"`
* Lowercasing all tokens
* Removing:

  * Tokens with fewer than 2 characters
  * Stopwords (`stopwords_en.txt`)
  * Words appearing only once in the corpus
  * Top 20 most frequent words by document frequency

### ✅ Output

* `processed.csv`: Cleaned review data
* `vocab.txt`: Alphabetically sorted vocabulary with index mappings (`word:index` format)

---

## 📊 2. Feature Representation

This stage involves transforming the pre-processed reviews into numerical representations suitable for machine learning models:

### Feature Types

* **Bag of Words (BoW)**: Count vector representation based on the processed vocabulary
* **Word Embeddings**: Using pre-trained language models (e.g., Word2Vec, GloVe, or FastText) to generate:

  * Unweighted average vectors
  * TF-IDF weighted average vectors

### ✅ Output

* `count_vectors.txt`: Sparse matrix representation of BoW features (`index:freq` format per line)

---

## 🤖 3. Review Classification

Machine learning models are built to classify whether a review recommends a product:

### Tasks

* **Language Model Comparison**: Evaluate performance of models using different feature representations
* **Information Gain Evaluation**: Compare model accuracy using:

  * Only review titles
  * Only review texts
  * Both title and review text

### Models Used

* Logistic Regression (primary)
* Other classification models (e.g., SVM, Random Forest, etc.) can be explored

### Evaluation Method

* 5-fold cross-validation to ensure robust performance comparisons

---

## 📂 Project Structure

```
📦 NLP-Clothing-Recommendation
│
├── data/
│   ├── reviews.csv
│   └── stopwords_en.txt
│
├── output/
│   ├── processed.csv
│   ├── vocab.txt
│   └── count_vectors.txt
│
├── embeddings/
│   ├── tfidf_embeddings.npy
│   └── avg_embeddings.npy
│
├── notebooks/
│   ├──  Text Pre-processing.ipynb          # Pre-processing
│   └── Generating Feature Representations for Clothing Reviews and classifcation.ipynb        # Feature extraction + Classification
│
└── README.md
```

---

## 💡 Key Skills Demonstrated

* Natural Language Preprocessing
* Feature Engineering for Text Data
* Word Embeddings and TF-IDF
* Binary Text Classification
* Model Evaluation and Analysis
* Jupyter Notebook Documentation

---

## 📌 Dependencies

* Python 3.8+
* Jupyter Notebook
* scikit-learn
* NumPy / pandas
* gensim / spaCy / nltk (depending on embedding model)
* matplotlib / seaborn (optional for visualization)

To install required packages:

```bash
pip install -r requirements.txt
```

---

## 📈 Results & Insights

The analysis explores:

* Which text representations work best for customer review classification
* How combining different types of information (title + review) impacts model performance
* How pre-trained embeddings compare with traditional BoW models in real-world scenarios

---

## 🔮 Future Improvements

* Integrating this NLP model with a web-based interface for live product recommendations
* Fine-tuning transformer models like BERT for better accuracy
* Exploring sentiment analysis for additional insights

---

## 📬 Contact

For questions, suggestions, or collaboration opportunities, feel free to reach out via GitHub Issues or email.

---

