# 🏦 Credit Risk Scoring & Loan Default Prediction

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/flask-3.0%2B-green.svg)
![XGBoost](https://img.shields.io/badge/xgboost-2.0.3-orange.svg)
![Bootstrap](https://img.shields.io/badge/bootstrap-5.3-purple.svg)

> **A production-ready Machine Learning web application to predict loan default probability with high accuracy.**

## 📌 Project Overview

This project is a **Credit Risk Assessment System** designed to help financial institutions and lenders make data-driven decisions. By analyzing applicant details such as income, credit history, and loan term, the application predicts the likelihood of loan default using an **XGBoost Classifier**.

The system features a **modern, responsive web interface** built with Flask and Bootstrap 5, providing real-time risk classification (Low, Medium, High) and actionable insights.

---

## 🚀 Key Features

- **✅ Real-Time Prediction**: Instant loan default probability calculation.
- **📊 Advanced ML Model**: Powered by **XGBoost** for superior accuracy and performance.
- **🛠️ Robust Preprocessing**: Automated handling of missing values, scaling, and categorical encoding using Scikit-Learn pipelines.
- **🎨 Modern UI/UX**: Clean, professional, and mobile-responsive design with gradient aesthetics.
- **🔒 Input Validation**: Secure form handling with both frontend and backend validation.
- **📈 Risk Visualization**: Color-coded risk levels and probability progress bars.

---

## 🏗️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python, Flask |
| **Machine Learning** | XGBoost, Scikit-Learn, Pandas, NumPy |
| **Frontend** | HTML5, CSS3, Bootstrap 5, JavaScript |
| **Data Processing** | Joblib, Pickle |
| **Environment** | Conda / Venv, Pip |

---

## 📂 Project Structure

```bash
credit-risk-scoring/
│
├── app.py                   # Main Flask application entry point
├── train_model.py           # ML pipeline: Data loading, training, evaluation, saving
├── generate_data.py         # Script to generate synthetic credit risk data
├── requirements.txt         # Project dependencies
├── README.md                # Project documentation
│
├── utils/
│   └── preprocessing.py     # Custom preprocessing pipeline (cleaning, scaling, encoding)
│
├── static/
│   ├── style.css            # Custom CSS for modern styling
│   └── script.js            # Frontend logic and validation
│
├── templates/
│   ├── index.html           # Main input form
│   ├── result.html          # Prediction result display
│   └── error.html           # Error handling page
│
└── model.pkl                # Trained model artifact (generated after training)
```

---

## ⚙️ How to Run Locally

Follow these steps to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/credit-risk-scoring.git
cd credit-risk-scoring
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Model
This step generates the synthetic dataset and trains the XGBoost model.
```bash
python train_model.py
```
*Output: You should see accuracy metrics and a confirmation that `model.pkl` has been saved.*

### 5. Run the Application
```bash
python app.py
```
*Output: `Running on http://127.0.0.1:5000`*

### 6. Access the App
Open your browser and navigate to:
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📸 Screenshots

### 1. Loan Application Form
*(Add screenshot of index.html here)*

### 2. Risk Assessment Result
*(Add screenshot of result.html here)*

---

## 🧠 Model Details

The model is trained on a synthetic dataset designed to mimic real-world credit risk scenarios. Key features include:

- **Credit Score**: The most significant predictor of default.
- **Debt-to-Income (DTI) Ratio**: Higher ratios generally indicate higher risk.
- **Annual Income & Loan Amount**: Used to assess affordability.
- **Employment Length**: Stability indicator.

**Performance Metrics:**
- **Accuracy**: ~85%
- **ROC-AUC**: ~0.92

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [Your Name]
