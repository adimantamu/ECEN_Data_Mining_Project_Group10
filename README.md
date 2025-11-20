# Instructions for Grading – Group 10 (KMNIST Project)

This repository contains the full implementation for Group 10’s KMNIST classification project.  
Follow the steps below to run and evaluate the final trained model.

---

## 1. Clone the repository

```bash
git clone https://github.com/adimantamu/ECEN_Data_Mining_Project_Group10.git
cd ECEN_Data_Mining_Project_Group10
```
---

## 2. Verify Python version

Make sure Python **3.9+** is installed:

```bash
python --version
```

---

## 3. Run the test script

The script automatically:

- installs missing packages  
- loads the saved model (`best_tuned_improved_cnn.pth`)  
- downloads KMNIST test data if needed  
- evaluates the model  
- prints metrics  
- displays or saves the confusion matrix  

Run:

```bash
python test.py
```

If a GPU is available, the script will use it automatically.

---

## 4. Expected Output

`test.py` will display:

- **Accuracy, Precision, Recall, F1-score**
- Full **classification report**
- **Confusion matrix**
    - shown on screen or saved as `confusion_matrix.png`
- A small **metrics summary table**

---

This setup requires no additional configuration. Everything should run as-is.
