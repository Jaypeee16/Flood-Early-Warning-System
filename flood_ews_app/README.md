# Flood Early-Warning System — Metro Manila Cities

CS 302 Modeling & Simulation Final Project · Technological Institute of the Philippines

## Overview

A Streamlit web application that predicts flood probability for four Metro Manila cities
(Manila, Marikina, Quezon City, Pasig) using a logistic regression model trained on a
sigmoid-augmented version of the *Metro Manila Flood Prediction 2016–2020* Kaggle dataset.

PAGASA-style four-tier colour warnings (Green / Yellow / Orange / Red) are issued based on
the predicted probability, using an optimal rainfall trigger threshold of **21.5 mm/day**
(10:1 false-negative to false-positive cost ratio).

## Project Structure

```
flood_ews_app/
├── app.py                          # Main Streamlit application
├── model/
│   ├── flood_model.joblib          # Trained model (optional — see Model Fallback)
│   └── model_coefficients.json     # Sigmoid fallback parameters
├── data/
│   └── city_profiles.csv           # Per-city adjustment factors
├── utils/
│   ├── prediction.py               # Probability computation
│   └── warning_tiers.py            # Tier classification
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Model Fallback Behaviour

The app tries to load `model/flood_model.joblib` at startup. If the file is absent,
it automatically falls back to the analytical sigmoid formula:

```
P(flood) = 1 / (1 + exp(−0.3 × (rainfall − 30)))
```

with parameters stored in `model/model_coefficients.json`. The app banner indicates
which mode is active. All other features (city adjustment, tier classification,
charts) work identically in both modes.

## Quick Test Checklist

| Input | Expected Result |
|---|---|
| 0 mm, any city | Green tier, ~0% probability |
| 21.5 mm, Manila | ~50% probability, Yellow/Orange boundary |
| 55 mm, any city | Red tier, >90% probability |
| Same rainfall, Marikina vs QC | Marikina shows ~15% higher probability |
