# MFCC-Based Speaker Identification

## Project Overview
This project focuses on **Speaker Identification** using audio recordings. It implements **Mel-Frequency Cepstral Coefficients (MFCC)** for feature extraction and uses **Machine Learning (SVM)** and **Convolutional Neural Networks (CNNs)** for classification. The system can identify speakers from audio samples by analyzing their voice characteristics.

MFCC is used to extract meaningful audio features that represent the timbre and frequency characteristics of the speaker's voice. These features are then fed into SVM and CNN models for accurate speaker identification.

---

## Dataset
We used the **Speaker Recognition Dataset** available on Kaggle:  
[Speaker Recognition Dataset](https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset)

> **Note:** The dataset is not included in this repository due to its size. You can download it from Kaggle and place it in the folder path specified in the scripts.

---

## Project Structure
mfcc_speaker_id/
│
├── notebooks/ 
│ ├── 1_ML_Model.ipynb 
│ ├── 2_Regression_Classification.ipynb # Notebook for regression & classification comparison
│ └── 3_CNN_Model.ipynb
│
├── README.md 
├── requirements.txt

## Features & Methods
### 1. MFCC Feature Extraction
- Extracts **13–40 MFCC coefficients** per audio file using `librosa`.
- Computes mean MFCC across time frames.
- Supports plotting waveform and MFCC spectrograms.

### 2. Machine Learning Models
- **Classification:** Support Vector Machine (SVM) trained to identify speakers.
- **Regression (optional):** SVM regression trained on speaker indices for comparison.
- Performance metrics: **Accuracy, Precision, Recall, F1-score**, and **Confusion Matrix**.

### 3. Convolutional Neural Network (CNN)
- Input: MFCC features reshaped for CNN.
- Architecture: Multiple convolutional + pooling layers, fully connected dense layers.
- Output: Softmax probabilities for speaker classes.
- Trained and validated with **train-test split** and evaluated with a **confusion matrix**.

---

## Usage

1. Install dependencies
pip install -r requirements.txt

3. Update dataset path in scripts
dataset_path = "c:/Users/shrav/archive/16000_pcm_speeches"

4. Run scripts
Machine Learning (SVM) Model:
python src/ml_model.py

Regression & Classification:
python src/regression_classification.py

CNN Model:
python src/cnn_model.py

Visualizations
Waveform plots: raw audio amplitude over time.
MFCC spectrograms: Frequency characteristics of the speaker’s voice.
Confusion matrices: model classification performance.
CNN training curves: Accuracy and loss over epochs.

Libraries Used
numpy, pandas – Data processing
librosa – Audio loading and MFCC extraction
matplotlib, seaborn – Visualizations
scikit-learn – SVM, metrics, train-test split
tensorflow / keras – CNN modeling

## References

Kaggle Speaker Recognition Dataset: https://www.kaggle.com/datasets/kongaevans/speaker-recognition-dataset
Librosa Documentation: https://librosa.org/doc/latest/index.html
Scikit-learn Documentation: https://scikit-learn.org/stable/

Notes
-Make sure to download the dataset from Kaggle before running the scripts.
-All scripts assume .wav files in the folder structure provided by the dataset.

## Author
Shravanthi Balasubramanian
Email: shravanthi.balasubramanian@gmail.com
