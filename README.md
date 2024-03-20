# Immo Prediction together with 🦀 Charlie 🦀
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)

![Immo House Predictions](./assets/charlie.png)

## 🏢 Description
My first Machine Learning project. Exciting! I use a dataset of houses that we scraped from the internet and in this repo 
I will apply Linear Regression together with Charlie 🦀 to predict the price of a house based on its features.

## 📦 Repo structure
```
├── assets
├── data
│   └── raw
│       └── data.csv  # the raw data
├── models  # the trained models
│   ├── basic_linearregression.pkl
│   ├── linearregression_log10.pkl
│   └── random_forest.pkl
├── README.md
├── requirements.txt
└── src
    ├── config.py
    ├── features
    │   ├── build_features.py  # add new features
    │   └── transformers.py  # transform features
    ├── models  # train models
    │   ├── model_utils.py 
    │   ├── pipeline.py  # base pipeline
    │   ├── train_basic_linearregression.py  
    │   ├── train_linearregression_log10.py
    │   └── train_random_forest.py
    └── utils.py
```

## 🚀 To retrain a model
Before charlie can predict the price of a house, we need to install the requirements.
```bash
# install requirements
pip install -r requirements.txt
```

Now Charlie is all set and ready to be trained. To train a model, run the following command in the terminal:
```bash
# change directory to source root folder
cd src

# train a model
python ./models/train_basic_linearregression.py # or the name of a different model
```
Charlie will print an R-squared score and save the model in the models folder with a similar name as the train_model.py 
file.
## Screenshot

## ⏱️ Timeline
This project was done in 4 days including studying the theory and implementing the code.

## 📌 Personal Situation
This project was done as part of my AI trainee program at BeCode.

### Connect with me!
[![LinkedIn](https://img.shields.io/badge/linkedin-%230077B5.svg?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/gerrit-geeraerts-143488141)
[![Stack Overflow](https://img.shields.io/badge/-Stackoverflow-FE7A16?style=for-the-badge&logo=stack-overflow&logoColor=white)](https://stackoverflow.com/users/10213635/gerrit-geeraerts)
[![Ask Ubuntu](https://img.shields.io/badge/Ask%20Ubuntu-dc461d?style=for-the-badge&logo=linux&logoColor=black)](https://askubuntu.com/users/1097288/gerrit-geeraerts)

