# Heart Failure Predictor

**Heart Failure Predictor** is a machine learning project designed to predict the likelihood of heart failure in patients with cardiovascular disease using a Support Vector Machine (SVM) model.

---

##  Project Overview

This project provides an end-to-end pipeline—from loading and processing patient data to training an SVM model, and finally serving predictions through a Streamlit web app.

###  Features

- **Data**: Uses clinical records data (`heart_failure_clinical_records_dataset.csv`).
- **Model**: Trained `heart_failure_model.h5` (SVM-based) for classification of heart failure risk.
- **Preprocessing**: `scaler.pkl` ensures consistent normalization of input features.
- **Interface**: A user-friendly web app built with Streamlit (`streamlit_app.py`) for interactive predictions.
- **Notebook**: `Cardio_vascular.ipynb` walks through data exploration, feature engineering, model building, and evaluation.

---

##  Getting Started

### Prerequisites

- Python 3.7 or above
- Recommended: Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```
### Clone the repository
```bash
git clone https://github.com/MahadhevanS/Heart-Failure-predictor.git
cd Heart-Failure-predictor
```
### Install Dependencies
```bash
pip install -r requirements.txt
```

### Try using the Streamlit web app 
```bash
streamlit run streamlit_app.py
```
