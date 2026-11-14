---
title: Credit Risk Model Demonstration
emoji: 📊
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
pinned: false
short_description: Historical credit-risk model with browser-local scoring
---

# Credit risk model demonstration

Jimi Umar · AAI6610 · Assignment 10.5

An educational deployment of the exact fitted Module 8 neural network and its preprocessing.
The app accepts synthetic profiles only. It is not a lending decision tool and does not assess
any real person's present-day creditworthiness.

## Model provenance

The saved pipeline `Module_08_Tuning/tuned_neural_network.joblib` was exported without training,
quantization or threshold tuning. `model.json` contains the fitted numeric medians, means and
scales, the learned one-hot categories, and all weights and biases of the 33-64-32-1 network.
The original joblib checksum is recorded in `manifest.json`. The release checks the JSON hash
before scoring. Never load untrusted pickle or joblib files.

Historical test accuracy is 81.90%, default precision 67.24%, recall 35.42%, F1 0.4640 and
ROC AUC 0.7670. The 6000-row test set was examined in earlier coursework. Export parity on
those records measures numerical runtime equivalence, not new independent model validation.
Fairness, calibration, present-day population validity and decision costs have not been established.

## Run and test

Serve this directory with any ordinary static HTTP server, for example `python -m http.server 8765`.
Open `http://localhost:8765`. Do not open index.html directly using file URLs, since module and
model loading require HTTP. A modern browser with Web Crypto is required; production uses HTTPS.

With Node.js 22 or later, run `node --test tests/*.test.js`. No npm dependencies are installed.
The expected scores in `fixtures.json` come from the original Python pipeline on 25 expressly
synthetic profiles. The tests also cover validation, imputation, field ordering, deterministic
repeat requests, malformed parameters and model-file tampering.

## Hosting and maintenance

Hugging Face Static Spaces serve the app. Inference runs on the visitor's CPU. This needs no
paid hosted Python runtime, GPU or training service. The browser fetches application assets,
not a remote prediction API. Profile values are not sent to a server or stored by this app.

GitHub Actions runs the tests on pull requests and main-branch pushes. A separate deployment
job depends on successful tests and syncs an explicit file allowlist to the Space. `HF_TOKEN`
is a GitHub Actions secret; `HF_SPACE_ID` is a repository variable. No credential belongs in
this repository. Use a token limited to writing the designated Space. A rollback restores a
previous complete application/model/manifest revision through the same tested workflow.

This repository contains fitted model parameters and synthetic examples, not the original
training data, original test rows, real client identifiers or the coursework report.

## Sources

- Yeh, I. (2009). Default of Credit Card Clients [Dataset]. UCI Machine Learning Repository.
  https://doi.org/10.24432/C55S3H (CC BY 4.0).
- Hugging Face. Spaces overview. https://huggingface.co/docs/hub/en/spaces-overview
- Hugging Face. Static HTML Spaces. https://huggingface.co/docs/hub/en/spaces-sdks-static
- Hugging Face. Managing Spaces with GitHub Actions. https://huggingface.co/docs/hub/en/spaces-github-actions
- Scikit-learn developers. Model persistence. https://scikit-learn.org/stable/model_persistence.html

The original dataset authors retain attribution. No endorsement by UCI or Hugging Face is implied.
