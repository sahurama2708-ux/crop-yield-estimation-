# 🌱 CropYield estimation (Deep Learning Edition)

An advanced, full-stack agricultural web application powered by a **Deep Learning Neural Network** designed to predict crop yields accurately based on multiple environmental, soil, and geographical parameters. Built with a modern dark-themed glassmorphic user interface.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-Framework-green?style=for-the-badge&logo=flask&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%2BLearning-orange?style=for-the-badge&logo=tensorflow&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-UI-38bdf8?style=for-the-badge&logo=tailwindcss&logoColor=white)

---

## ✨ Key Features

- **🔐 User Session & Authentication:** Personalized portal login system mapped with user name and email.
- **⚡ Deep Learning Prediction Engine:** Utilizes a trained Sequential Neural Network model (`.h5`) combined with feature mapping (113 features) and standard scaling for precise quintal-per-acre estimations.
- **📜 Secure History Tracking:** Automatically logs past predictions into an embedded SQLite database (`history.db`) tied directly to the user's email.
- **📥 Report Generation:** Instant text-based report download feature containing all predictive parameters and estimated outputs.
- **🎨 Glassmorphic Agricultural UI:** Immersive dark-mode interface styled with forest greens, neon accents, and smooth backdrop-blur effects.

---

## 🛠️ Tech Stack

* **Frontend:** HTML5, Tailwind CSS, JavaScript (Vanilla), Google Fonts (`Plus Jakarta Sans`)
* **Backend:** Python, Flask Microframework
* *deep learning / AI:** TensorFlow, Keras, Scikit-Learn, Joblib, NumPy, Pandas

-------------------------------
  run vs codes commands::
  cd "crop-yield-estimation"
  
  pip install -r requirements.txt
  
  pip install flask tensorflow pandas numpy joblib scikit-learn
  
  pip install flask tensorflow pandas numpy joblib scikit-learn
-----------------------------------------------
---

## 📁 Project Directory Structure

```text
crop-yield-estimation/
│
├── app/
│   ├── app.py              # Flask Backend & SQLite Database Routes
│   ├── history.db          # User Prediction History Database
│   └── templates/
│       └── index.html      # Glassmorphic Frontend Dashboard UI
│
├── artifacts/
│   ├── ultimate_crop_yield_dl_model.h5  # Deep Learning Model Weights
│   ├── scaler.pkl          # Feature Standard Scaler
│   └── feature_columns.pkl # 113 Processed Feature Columns Mapping
│
├── requirements.txt        # Python Dependencies
└── README.md               # Project Documentation
