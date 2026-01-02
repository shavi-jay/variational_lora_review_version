## Variational Low-Rank Adaptation for Uncertainty Quantification in Molecular Property Prediction
Authors: *I. Shavindra Jayasekera, Jochen Sieg, Miriam Mathea, Yingzhen Li*

This is the codebase for the Journal of Cheminformatic submission: "Variational Low-Rank Adaptation for Uncertainty Quantification in Molecular Property Prediction".


### Create Environment
#### Prerequisites (verify/install Python 3.10)

Check Python 3.10:

```
python3.10 --version
```


If missing (Ubuntu/Debian example):

```
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv
```

#### 1) Set up venv on Linux

Create the virtual environment named `variational_lora`:

```
python3.10 -m venv variational_lora
```

#### 2) Activate the variational_lora environment (Python 3.10)

```
source variational_lora/bin/activate
python --version  # should show Python 3.10.x
```

#### 3) Install libraries from requirements.txt via pip

Upgrade pip (optional but recommended):

```
python -m pip install --upgrade pip
```

Install dependencies:

```
pip install -r requirements.txt
```

Deactivation (when done)
```
deactivate
```