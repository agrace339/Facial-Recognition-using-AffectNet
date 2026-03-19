# Project Setup and Usage

ADD DESCRIPTION OF PROJECT

---

## 1. Clone the Repository

```bash
git clone https://github.com/agrace339/Facial-Recognition-using-AffectNet.git
cd Facial-Recognition-using-AffectNet
```

---

## 2. Create a Python Virtual Environment

It's recommended to use a virtual environment to isolate dependencies.

```bash
# Create a virtual environment named "venv"
python -m venv venv

# Activate the environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

---

## 3. Install Required Packages

Install all Python dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

**Required Packages:**

- `numpy`
- `pandas`
- `Pillow`
- `tqdm`
- `kagglehub`

> All versions are specified in `requirements.txt` to ensure reproducibility.

---

## 4. Run the Scripts

All scripts are located in the `scripts/` folder. You need to run them in the following order:

### **Step 1: Download and organize raw dataset**

```bash
python script/1_data_import.py
```

- Downloads the AffectNet dataset from Kaggle using `kagglehub`.
- Organizes files into `data/raw/images` and `data/raw/labels`.
- Cleans temporary cache used by `kagglehub`.

### **Step 2: Clean and process images**

```bash
python script/2_data_cleaning.py
```

- Removes corrupt, blurry, overexposed, or duplicate images.
- Converts images to grayscale and resizes them to 96x96 pixels.
- Saves cleaned images to `data/cleaned/images`.
- Generates a CSV annotation file `data/cleaned/affectnet_annotations.csv`.

### **Step 3: Principal Component Analysis**

```bash
python script/PCA_3.py
```

- Defines PCA Class which applies PCA to a Training set to get a set of principal components which can be applied to images to project them onto a smaller feature space.
- Generates a scatter plot of the two most important components
- Generates an animation of a reconstuction of one of the training images

### **Step 4: SVM Analysis**

```bash
python script/SVM_4.py
```

### **Step 5: Ensemble method and random forest Analysis**

```bash
python script/5_Ensemble_Method.py
```

## 6. Directory Structure

After setup and running scripts, your folder structure should look like this:

```
project-root/
│
├─ data/
│  ├─ raw/
│  │  ├─ images/
│  │  └─ labels/
│  └─ cleaned/
│     ├─ images/
│     │  ├─ train/
│     │  ├─ val/
│     │  └─ test/
│     └─ affectnet_annotations.csv
│
├─ scripts/
│  ├─ 1_data_import.py
│  ├─ 2_data_cleaning.py
│  └─ PCA_3.py
│  ├─ SVM_4.py
│  └─ 5_Ensemble_Method.py
│  └─ randomforest.py
│
├─ requirements.txt
└─ README.md
```

