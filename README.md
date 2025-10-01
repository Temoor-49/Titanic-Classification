# 🛳️ Titanic Survival Classification

## 📌 Overview
This project predicts passenger survival on the Titanic using machine learning.  
We preprocess the dataset, perform exploratory data analysis (EDA), engineer new features, train multiple models, and evaluate their performance.  
The best model (Random Forest) was selected based on validation accuracy and used for predictions.

---

## 📂 Project Structure
my-first-ds-project/
├─ data/ # raw and processed data
├─ notebooks/ # exploratory notebooks & plots
│ └─ eda.ipynb
├─ src/ # source code
│ ├─ init.py
│ ├─ data.py
│ ├─ features.py
│ ├─ model.py
│ ├─ train.py
│ └─ predict.py
├─ models/ # saved models (.joblib)
├─ predictions.csv # sample predictions
├─ environment.yml # conda environment
├─ README.md # project documentation
└─ .gitignore

yaml
Copy code

---

## ⚙️ Setup Instructions

### 1. Clone & Environment
```bash
git clone https://github.com/<your-username>/titanic-classification.git
cd titanic-classification
conda env create -f environment.yml
conda activate ds101
2. Download Data
bash
Copy code
kaggle competitions download -c titanic -p data/
unzip data/titanic.zip -d data/
3. Train Model
bash
Copy code
python src/train.py
4. Predict
bash
Copy code
python src/predict.py
📊 Exploratory Data Analysis
Some key findings:

Sex: Women had much higher survival rates than men.

Pclass: First-class passengers were more likely to survive than third-class.

Age: Children had higher chances of survival than older passengers.

Family Size: Families of 2–4 had better survival than solo travelers.

(Plots available in notebooks/eda.ipynb and notebooks/plots/)

🤖 Models Compared
Logistic Regression → Accuracy: 0.8045

Random Forest → Accuracy: 0.8268 ✅

Gradient Boosting → Accuracy: 0.8268

Best model: Random Forest
Saved in: models/random_forest.joblib

📝 Results
Validation Accuracy: 82.68%

Kaggle Public Score: 0.76794

🔄 Reproducibility
Export environment for others:

bash
Copy code
conda env export > environment.yml
📌 Next Steps
Hyperparameter tuning (GridSearchCV/RandomizedSearchCV)

Try XGBoost / LightGBM for better accuracy

Add feature engineering from Cabin/Ticket columns

✍️ Author: Temoor Hussain

pgsql
Copy code

---

