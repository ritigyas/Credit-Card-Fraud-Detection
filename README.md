# Credit Card Fraud Detection Using Isolation Forest

## Project Description
This project detects fraudulent transactions from a large dataset of credit card transactions using **Isolation Forest**, an unsupervised anomaly detection algorithm. The model classifies transactions as either:
- **Legitimate (0)**  
- **Fraudulent (1)**  

The goal is to build a robust fraud detection system that identifies fraudulent transactions while minimizing false positives.

---

## Dataset
The project uses the **Credit Card Fraud Detection Dataset** from **Kaggle**, which contains **284,807 transactions** with the following features:
- **Features:**  
   - `V1` to `V28`: PCA-transformed features representing transaction details  
   - `Time`: Seconds elapsed between transactions  
   - `Amount`: Transaction amount  
- **Target label:**  
   - `Class`:  
     - `0` → Legitimate transactions  
     - `1` → Fraudulent transactions  
     
The dataset has a **highly imbalanced distribution**:  
- **Legit transactions:** 284,315 (~99.83%)  
- **Fraud transactions:** 492 (~0.17%)  

---

## Technologies Used
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas`: Data manipulation and analysis  
  - `numpy`: Numerical operations  
  - `sklearn`: Machine learning models and evaluation metrics  
  - `os`: File management  
- **Model:**  
  - `Isolation Forest` → Anomaly detection algorithm  

---

## Modeling Steps

### 1. Data Preprocessing
- The dataset is loaded using `pandas` from a `.csv` file.  
- The **features (`X`)** and **labels (`y`)** are separated.  

### 2. Contamination Rate Calculation
- The model uses the **real fraud rate** (`0.0017`) as the contamination rate, preventing overestimation of anomalies.  

### 3. Model Training
- The Isolation Forest model is trained using:  
   - `contamination=0.0017` → Fraud rate in the dataset  
   - `random_state=42` → Ensures reproducibility  
- Predictions are made, and the anomalies are labeled:  
   - `1 → Legit`  
   - `-1 → Fraud`  
- The labels are correctly mapped to:  
   - `0 → Legit`  
   - `1 → Fraud`  

### 4. Model Evaluation
- The results are saved in `data/anomaly_results_corrected.csv`.  
- The model’s performance is evaluated using:  
   - **Confusion Matrix**  
   - **Classification Report** (Precision, Recall, F1-score)  

---

## Results
After running the model:
- The output file `anomaly_results_corrected.csv` contains all transactions with the additional `Anomaly` column indicating fraud predictions.  
- The **Confusion Matrix** and **Classification Report** display the model’s accuracy and effectiveness.  

---

## Future Enhancements
- Feature Engineering: Add more engineered features to improve accuracy.
- Resampling Techniques: Use SMOTE or undersampling to balance the dataset.
- Visualization: Add data visualizations to better interpret the results.

