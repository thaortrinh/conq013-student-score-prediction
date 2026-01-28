# CONQ013 Project  
# Student Score Prediction

This project focuses on **predicting student scores** using machine learning techniques.  
It is designed as a learning-oriented ML project, covering environment setup, data organization,
and model training.

### Dataset
The dataset is obtained from Kaggle:

🔗 https://www.kaggle.com/competitions/playground-series-s6e1/data

---

## 1. Environment Setup

### 1.1. Create a virtual environment

#### Windows (Command Prompt / PowerShell)
```bat
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS / Git Bash / WSL
```bat
python3 -m venv venv
source venv/bin/activate
```

After activation, your terminal should show:
```text
(venv)
```

### 1.2. Install dependencies
```bat
pip install -r requirements.txt
```


### 2. Data Setup

Create the following folder structure in the project root:
```
data/
├── raw/
├── interim/
└── processed/

```
**Steps:**

Create a data folder in the project root.

Inside data, create 3 subfolders:

- raw/ – for original, unprocessed data

- interim/ – for cleaned data

- processed/ – for train-ready data


Download the dataset from Kaggle.

Place the file train.csv into:
```
data/raw/train.csv
```

## 3. Run code
### 3.1. Running the Jupyter Notebook
Navigate to the notebooks directory: 
```
cd notebooks
```
Launch Jupyter Notebook:
```
jupyter notebook
```