# 🫀 Heart Disease Prediction with Personalized Diet Plan
![Status](https://img.shields.io/badge/Status-Active-yellow)
![Accuracy](https://img.shields.io/badge/Accuracy-94%25-blue)
![Language](https://img.shields.io/badge/Language-Python-blue)
![Framework](https://img.shields.io/badge/Framework-sklearn-yellow)
![License](https://img.shields.io/badge/License-MIT-blue)
![Data](https://img.shields.io/badge/Data-Clinical-yellow)

This project presents an advanced Heart Disease Prediction System coupled with a sophisticated Personalized Diet and Lifestyle Recommendation Engine. Leveraging state-of-the-art machine learning models, it accurately assesses an individual's risk of heart disease based on comprehensive clinical and lifestyle parameters. Beyond just prediction, the system intelligently crafts a tailored diet designed to mitigate identified risks and promote cardiovascular wellness.

---

![Project Banner](/assets/banner.jpg) <!-- Replace with your actual banner path -->

---

## 📌 Overview
This project is a **Heart Disease Prediction System** integrated with a **Personalized Diet Recommendation Engine**.  
It uses machine learning models to predict the likelihood of heart disease from clinical and lifestyle parameters, and then suggests a customized diet plan based on the risk level.

The project also saves trained models inside the `saved_model/` directory for reusability and performs several preprocessing and vital operations behind the scenes.

---

## 🛠️ Technologies & Libraries
- Python: The core programming language.
- Scikit-learn: For machine learning model development and evaluation.
- Pandas & NumPy: For data manipulation and analysis.
- Matplotlib & Seaborn: For data visualization.
- Jupyter Notebook: For interactive development and the user interface.

---

## 🚀 Features
- Heart disease risk prediction using ML models.
- Personalized vegetarian diet plan suggestions.
- Saves trained models inside `saved_model/` for later use.
- Interactive interface built with Jupyter Notebook (`Interface.ipynb`).
- Clean dataset included (`balanced_heart_disease.csv`).
- Modular structure for easy understanding and extension.

---

## 📂 Project Structure
```bash
├── balanced_heart_disease.csv   # Processed dataset used for training & testing
├── Diet_chart.ipynb             # Diet recommendation logic & chart generation
│   └── Final_diet_plan.json     # Generated final diet plan in JSON format
├── Heart.ipynb                  # Core ML model training & evaluation
├── Interface.ipynb              # User-facing interface for predictions
├── saved_model/                 # Directory where trained models are stored
│   ├── feature_info_20250906_075310.pkl       # Feature information
│   ├── heart_disease_model_20250906_075310.pkl # Trained heart disease model
│   ├── label_encoders_20250906_075310.pkl     # Saved label encoders
│   ├── model_info_20250906_075310             # Model info file
│   └── model_metadata_20250906_075310.pkl     # Metadata of trained model
├── assets/                      # Static resources (images, icons, etc.)
│   └── banner.jpg               # Project banner image
└── README.md                    # Project documentation
```
---

# ⚙️ Installation & Setup
 ##   1️⃣ Clone the repository
   ```bash
   git clone https://github.com/Soumen633/Heart-DiseasePrediction-With-Personalized-Diet-Plan
   cd Heart-DiseasePrediction-With-Personalized-Diet-Plan
   ```
 ## 2️⃣ Create a virtual environment (optional but recommended)
 ```bash
 python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

## 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
## 4️⃣ Run Jupyter Notebook
```bash
jupyter notebook
```
---
# 🛠️Extra Instructions

- Heart.ipynb → to train/evaluate model
- Diet_chart.ipynb → to generate personalized diet plans
- Interface.ipynb → to interact with the system
---
# ▶️ Usage
- Train the ML model using Heart.ipynb.

- A trained model will be saved automatically inside the saved_model/ directory.

- Run Interface.ipynb for an interactive prediction interface.

- Based on the prediction, use **Diet_chart.ipynb** to view the recommended diet plan.

---
## 👥 Contributors

This project was developed as a **group project** by:

- [@Soumen633](https://github.com/Soumen633) — *Soumen Nandi*  
- [@P-Rawani001](https://github.com/P-Rawani001) — *Pankaj Kumar Rawani*  
- [@AMRITA-2002](https://github.com/AMRITA-2002) — *Amrita Mandal*  



