# 💧 Water Potability Prediction System

A Machine Learning project that predicts whether water is **drinkable (potable)** or **not drinkable** based on its chemical properties.  
This project includes **data preprocessing, model training, evaluation, hyperparameter tuning, and deployment** with a **Gradio web interface** hosted on Hugging Face Spaces.

---

## 📂 Project Structure

```

├── train.py                    # Model training, evaluation, and saving
├── app.py                      # Gradio web interface for predictions
├── water_predict.csv           # Dataset (Water Potability dataset)
├── water_predict_model.pkl     # Saved trained pipeline
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation

```


---

## ⚙️ Steps Implemented

1. **Data Loading** – Loaded dataset and verified shape.  
2. **Data Preprocessing** –  
   - Handled missing values  
   - Outlier detection & removal (IQR method)  
   - Feature scaling (Standardization)  
   - Feature engineering (`quality_index`)  
   - Train-test split  
3. **Pipeline Creation** – Integrated preprocessing + model.  
4. **Model Selection** – Chose **Random Forest Classifier** for robustness and interpretability.  
5. **Model Training** – Trained pipeline on training data.  
6. **Cross-Validation** – 5-fold CV with mean ± std reporting.  
7. **Hyperparameter Tuning** – GridSearchCV for best parameters.  
8. **Best Model Selection** – Selected final tuned pipeline.  
9. **Model Evaluation** – Accuracy, precision, recall, F1-score, confusion matrix.  
10. **Gradio Web Interface** – User-friendly interface for predictions.  
11. **Deployment** – Hosted on Hugging Face Spaces.

---

## 🚀 How to Run Locally

1. **Clone the repo:**

   ```bash
   git clone https://github.com/<your-username>/water-potability-prediction.git
   cd water-potability-prediction


**2. Install dependencies:**

```
pip install -r requirements.txt

```

**3. Train the model:**
```
python train.py
```

**4. Launch the Gradio app:**

```
python app.py
```

---

# 🌐 Hugging Face Deployment
This project is deployed on Hugging Face Spaces with Gradio.
👉 Live Demo: https://huggingface.co/spaces/rubina25/Water-Potability-Prediction

---

# 📊 Example Input & Output

| pH   | Hardness | Solids | Chloramines | Sulfate | Conductivity | Organic Carbon | Trihalomethanes | Turbidity | Prediction      |
|------|----------|--------|-------------|---------|--------------|----------------|-----------------|-----------|----------------|
| 7.0  | 200      | 15000  | 7.0         | 350     | 400          | 10             | 80              | 3.0       | ❌ Not Drinkable|
| 3.5  | 100      | 25000  | 2.0         | 150     | 200          | 20             | 20              | 5.0       | ✅ Drinkable   |
| 8.2  | 180      | 12000  | 6.5         | 400     | 450          | 8              | 90              | 2.5       | ❌ Not Drinkable|
| 4.0  | 90       | 30000  | 1.5         | 100     | 150          | 25             | 15              | 6.0       | ✅ Drinkable |


# 🛠️ Tech Stack

- Python

- Pandas, NumPy, Scikit-learn

- Gradio (Web Interface)

- Hugging Face Spaces (Deployment)

---

# 📌 Future Improvements

- Add more advanced models (XGBoost, LightGBM).

- Improve feature engineering with domain knowledge.

- Add visualization dashboards.


---

# 👨‍💻 Author

Developed by **Rubina Begum** 

Feel free to connect and explore more projects!
