# 🌸 Iris Flower Species Classifier (End-to-End ML Deployment)

An end-to-end Machine Learning web application built using Python, Scikit-learn, and Streamlit. The system implements an industry-standard directory structure to ingest tabular morphological data, split and train an ensemble classifier, and deploy a real-time reactive user interface dashboard.

---

## 🚀 Live Demo & Visuals

*(Optional: Once you take your project video, you can upload a screenshot or a short GIF here to show your live dashboard in action!)*

---

## 🛠️ Tech Stack & Architecture

* **Programming Language:** Python 3.10+
* **Data Manipulation:** Pandas
* **Machine Learning Framework:** Scikit-learn (Random Forest Classifier)
* **Web Dashboard & UI:** Streamlit
* **Version Control:** Git & GitHub

### Project Directory Structure
```text
iris-classification-project/
│
├── data/
│   └── Iris.csv              # Raw tabular dataset containing morphological features
│
├── models/
│   └── iris_model.pkl        # Serialized, trained Random Forest model object
│
├── src/
│   └── train.py              # Modular Python pipeline script for cleaning & model fitting
│
├── app.py                    # Real-time Streamlit dashboard application file
├── .gitignore                # Restricts system dependencies/virtual environments from tracking
└── requirements.txt          # Explicit list of version-controlled packages needed
