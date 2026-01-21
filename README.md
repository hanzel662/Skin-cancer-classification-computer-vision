# Classification of Skin Cancer Using XGBoost and SVM Ensemble Learning Based on Dermoscopy Image Features

## 👨‍💼 Author
* Richard Chrysander - 2702242972
* Hanzel Octavian Lesmana - 2702241162
* Felix Stevanus - 2702252090

## 📂 Project Overview
This project focuses on the classification of skin cancer using dermoscopy images from HAM10000 dataset through a feature-based machine learning approach. The main objective is to develop an accurate and computationally efficient classification system that can assist medical professionals in diagnosing skin cancer more objectively and consistently.

## 🛠️ Configuration
All training and experiment hyperparameters are stored directly in `init_model.py`, determined using RandomizedSearchCV

## 🖥️ Model & Approach
This project uses a feature-based ensemble learning approach for skin cancer classification.

- **Dataset**: Skin Cancer MNIST: HAM10000 ([Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000))
- **Model**: SVM + XGBoost Ensemble Learning Model with Soft Voting Strategy

Metrics:
* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

## 💽 How to Run Demo Application
* Clone the Repository:
  `git clone https://github.com/hanzel662/Skin-cancer-classification-computer-vision.git`
* Open the project folder
* Enter the app folder:
   `cd app`
* Install dependencies:
  `pip install -r requirements.txt`
* Run the application:
  `streamlit run app.py`
* Application should have opened automatically.

## 💯 Results
Training logs, plots, and saved models are stored in `output/`