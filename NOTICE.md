# Notices and Attribution

Project: Bank Customer Churn Risk Scorer

This repository contains source code, experiment scripts, model-training utilities, model-comparison scripts, and documentation for a lightweight bank customer churn prediction project.

## Project Purpose

This project is a portfolio and educational demonstration of minimally sized, browser-deployable machine learning. It compares compact PyTorch artificial neural networks against standard tabular baselines and selects a small ANN for static, local browser inference.

The selected deployment direction is not based on maximizing offline benchmark accuracy alone. It is based on a deliberate tradeoff: accepting a small performance loss relative to the strongest tabular benchmark in exchange for a smaller, simpler model that can run without backend inference, API keys, or paid cloud services.

## Code License

Unless otherwise noted, source code in this repository is licensed under the Apache License, Version 2.0. See `LICENSE` for details.

## Documentation and Figures

Documentation, README content, diagrams, figures, and model-card style materials may be distributed under CC BY 4.0 if separately marked or released that way. If no separate documentation license is provided, consult the repository owner before reuse beyond normal GitHub viewing and citation.

## Model Weights and Trained Artifacts

Trained model weights, checkpoints, exported model files, preprocessing schemas, and deployment manifests are research/demo artifacts. Recommended terms for these artifacts are CC BY-NC 4.0 or explicit research/demo-use terms, unless a separate model artifact license is added.

The model artifacts are not intended for production banking, lending, retention, eligibility, or consumer decision-making use without independent validation, security review, fairness analysis, monitoring, and compliance review.

## Dataset Attribution

This project uses the Bank Customer Churn dataset accessed through Maven Analytics Data Playground.

Dataset: Bank Customer Churn  
Provider / access point: Maven Analytics Data Playground  
Source page: https://mavenanalytics.io/data-playground/bank-customer-churn  
Direct download: https://maven-datasets.s3.amazonaws.com/Bank+Customer+Churn/Bank+Customer+Churn.zip  
Original source listed by Maven: Kaggle  
License listed by Maven: Public Domain  
Access date: 2026-05-19

The dataset is third-party data. Dataset rights and reuse are governed by the original dataset license and terms, not by this repository's code license.

## Third-Party Dependencies

This project may use third-party Python packages, including but not limited to:

- PyTorch
- NumPy
- pandas
- scikit-learn
- XGBoost
- joblib
- ONNX / ONNX Runtime, if exporting or deploying ONNX models

Each dependency is governed by its own license. See the dependency metadata, package distribution, or project homepage for exact terms.

## Responsible-Use Notice

This repository is not a production banking system. It is an educational and portfolio demonstration of lightweight predictive modeling and browser-oriented deployment.

Before real-world use, any model trained from this repository should be evaluated for calibration, subgroup performance, fairness, data drift, security, privacy, and applicable banking, consumer-protection, and data-governance requirements.
