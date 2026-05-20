# Responsible Use Disclaimer

## Purpose of This Project

This project is a portfolio and educational demonstration of lightweight, browser-deployable machine learning. It shows how a compact predictive model can be trained offline and deployed as a static local-inference web app without backend infrastructure, API keys, or paid cloud inference.

The project is not a production banking system.

## Not Intended for Production Decision Making

The model is **not intended for real-world production decision making** without additional validation, monitoring, security review, fairness analysis, privacy review, and domain-specific compliance review.

Do not use this model as the sole basis for:

- banking decisions;
- credit or lending decisions;
- eligibility determinations;
- pricing decisions;
- account restrictions;
- adverse customer treatment;
- automated customer segmentation in a real institution;
- any decision that materially affects a person's access to financial products or services.

## Appropriate Demonstration Uses

Appropriate uses include:

- portfolio demonstration;
- educational explanation of binary classification;
- lightweight ML deployment demonstration;
- comparison of compact ANN models against tabular baselines;
- static browser inference prototype;
- discussion of deployment tradeoffs between raw accuracy and minimality.

## Human Oversight

If adapted for any real decision-support context, predictions should be reviewed by qualified human stakeholders. A churn-risk score should be treated as one signal among many, not as a final decision.

Users should understand:

- what the model predicts;
- what data was used;
- what the model does not know;
- what uncertainty and calibration limitations exist;
- what actions are appropriate based on the prediction.

## Fairness and Bias

The current project has not completed a formal fairness analysis.

Before real-world use, evaluate model performance across relevant subgroups. Depending on the dataset and jurisdiction, this may include analysis by geography, age bands, gender categories, account characteristics, or other legally and ethically relevant groups.

Recommended checks:

- subgroup ROC-AUC;
- subgroup precision/recall;
- subgroup calibration;
- false-positive and false-negative rates;
- impact of using top-risk targeting thresholds;
- sensitivity to missing or unknown categorical values.

## Calibration and Uncertainty

The model outputs a probability-like churn-risk score, but probability calibration should be separately validated before operational use.

Recommended checks:

- reliability diagrams;
- Brier score by subgroup;
- calibration curves;
- threshold sensitivity;
- comparison with business costs of false positives and false negatives.

## Data Drift and Monitoring

A real churn model would require ongoing monitoring. Customer behavior, product offerings, economic conditions, and retention strategies can change over time.

Recommended monitoring:

- feature distribution drift;
- prediction distribution drift;
- churn-rate drift;
- calibration drift;
- metric decay over time;
- retraining criteria;
- review of edge cases and user feedback.

## Privacy and Security

The intended deployment is client-side browser inference. This has advantages and tradeoffs.

Potential advantages:

- no customer data needs to be sent to a backend for inference;
- no API keys are required;
- no paid cloud inference service is required.

Potential risks:

- model artifacts are visible to users;
- preprocessing logic is visible to users;
- client-side code can be inspected or modified;
- sensitive input handling still requires careful UI and storage design;
- browser deployments should avoid storing customer data unless explicitly needed and properly protected.

For a public demo, use synthetic/example inputs rather than real customer records.

## Compliance Considerations

Banking and financial applications may be subject to laws, regulations, and institutional policies. Requirements vary by jurisdiction and use case.

Before any real-world use, consult qualified legal, compliance, security, and domain experts.

Potential review areas:

- privacy law;
- consumer-protection rules;
- anti-discrimination rules;
- model risk management;
- explainability and auditability;
- data retention policies;
- customer communication policies.

## Recommended User-Facing Disclaimer

A deployed demo should include a visible disclaimer such as:

```text
This demo is for educational and portfolio purposes only. It estimates churn risk from sample customer/account inputs using a lightweight machine learning model. It is not intended for production banking decisions, credit decisions, or automated customer treatment. Real-world use would require additional validation, fairness analysis, monitoring, security review, and compliance review.
```

## Summary

This project demonstrates that useful predictive ML can be small, local, and browser-deployable. The model is appropriate as an educational artifact and deployment prototype, not as a validated production decision system.
