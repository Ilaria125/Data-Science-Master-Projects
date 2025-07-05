# 🩺📰 Fake vs Real Data Detection

## Overview

This project was developed for the **Machine Learning module** during the MSc in Data Science at the University of Barcelona. The goal was to **detect whether data samples are fake or real**, distinguishing human-generated data from LLM-generated data across two domains:

- **Dataset A (Health Domain)**: Skin lesion characterization data.
- **Dataset B (Text Domain)**: News headlines.

The competition required **building classification models to predict fake vs real samples** and preparing a submission concatenating the predictions for Dataset A (256 samples) followed by Dataset B (1050 samples), for a total of 1306 predictions.

## Project Objectives

- Perform **Exploratory Data Analysis (EDA)** to understand feature distributions and identify data issues.
- Build **data cleaning and preprocessing pipelines** for structured and text data.
- Train and evaluate **classification models** using scikit-learn and XGBoost.
- Apply **NLP preprocessing and feature engineering** for text data.
- Optimize models using cross-validation with **accuracy as the primary metric**.
- Generate submission files following the competition format.

## Repository Structure

- `datasetA_model.ipynb`: Notebook for Dataset A (Health Domain), including EDA, preprocessing, model training, evaluation, and prediction generation.
- `DatasetB_model.ipynb`: Notebook for Dataset B (Text Domain), including text preprocessing, feature engineering (TF-IDF, embeddings), model training, evaluation, and prediction generation.

## Technologies Used

- **Languages:** Python
- **Libraries:** pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn, nltk, spaCy
- **Tools:** Jupyter

## Outcome

Successfully developed a complete pipeline to classify fake vs real samples, achieving competitive accuracy in both domains while practicing practical machine learning workflows aligned with real-world competition settings.

## Author

Ilaria Curzi – [ilaria2.curzi@gmail.com](mailto:ilaria2.curzi@gmail.com)

---

*This project is part of my portfolio for the MSc in Data Science at the University of Barcelona.*

