# App Review Analytics for Health Insurance Apps

This repository contains the data preparation, exploratory analysis, model training, inference, and dashboard integration workflow for a Master's thesis project on app store review analytics. The project focuses on German-language and multilingual user reviews of health insurance apps, with the goal of identifying recurring review topics and making them accessible in an interactive dashboard.

## Project goal

The project builds an end-to-end NLP pipeline for app reviews. It starts with review extraction and preprocessing, continues with manual labeling and exploratory topic discovery, and ends with supervised multi-label topic classification and dashboard-ready prediction exports.

The central objective is to support structured analysis of user feedback from app stores. Instead of reading reviews individually, the pipeline makes it possible to detect recurring issue categories such as registration problems, technical instability, document management, or electronic patient record features.

## Repository structure

The workflow is organized into a sequence of notebooks and supporting Python modules.

### Notebooks

- `01_app_reviews_hek_viactiv.ipynb`  
  Extracts and consolidates app review data, performs initial cleaning, and creates the base review dataset.

- `02_create_manual_labeling_subset_hek_viactiv.ipynb`  
  Creates a manual labeling subset and a remaining training pool for later supervised modeling.

- `03_exploratives_multitopic_clustering_reviews_final_en.ipynb`  
  Performs exploratory BERTopic-based clustering with multilingual embeddings to identify potential topic structures in the review corpus.

- `04_model_training_multilingual_vs_german.ipynb`  
  Compares Transformer-based multi-label topic classification models on the manually labeled review subset.

- `05_model_training_approach_loss_experiment.ipynb`  
  Runs the final model selection experiment, comparing joint multi-label classification vs. one-vs-rest as well as standard BCE vs. weighted BCE.

- `06_model_training_inference_evaluation.ipynb`  
  Applies the selected final model, generates evaluation artifacts for documentation, and exports dashboard-ready prediction files.

### Python modules

Depending on the current development stage, the repository may also contain supporting modules such as:

- `review_fetchers.py` for review collection
- `preprocessing.py` for review cleaning and normalization
- `clustering.py` for exploratory topic modeling support
- `inference.py` for prediction logic
- `aggregation.py` for dashboard-level aggregations
- `charts.py` for visual outputs
- `app.py` for the Streamlit prototype

## End-to-end workflow

The project follows six main stages:

1. **Review collection and preparation**  
   Reviews are collected from app sources, standardized, cleaned, and consolidated into a common dataset.

2. **Manual labeling subset creation**  
   A manageable subset is sampled for human annotation, while the remaining reviews are retained as a future training or inference pool.

3. **Exploratory topic discovery**  
   BERTopic and multilingual sentence embeddings are used to identify candidate themes and multi-topic structures in the review corpus.

4. **Supervised model training**  
   Manually labeled reviews are transformed into a multi-label classification dataset and used to fine-tune Transformer models.

5. **Final model selection**  
   Alternative modeling approaches and loss functions are compared under cross-validation and calibrated threshold strategies.

6. **Inference and dashboard export**  
   The selected production candidate is applied to review data and the final output is prepared for use in the Streamlit dashboard.

## Data flow

The pipeline distinguishes between several dataset roles:

- **Raw extracted reviews**: consolidated app review data after collection
- **Manual labeling subset**: small, human-annotated sample used for supervised learning
- **Training pool**: remaining reviews not initially labeled manually
- **Clustered exploratory outputs**: topic discovery artifacts from BERTopic
- **Model predictions**: per-label probabilities and binary predictions for dashboard use

A typical data flow looks like this:

`raw reviews -> cleaned review dataset -> manual labeling subset -> labeled training data -> trained classifier -> dashboard-ready predictions`

## Modeling approach

The supervised modeling task is formulated as a **multi-label text classification** problem. A single review can express more than one issue at the same time, so the model predicts multiple labels independently rather than forcing each review into exactly one class.

Across the modeling notebooks, the project experiments with:

- multilingual and German-capable Transformer backbones
- joint multi-label classification
- one-vs-rest classification
- standard `BCEWithLogitsLoss`
- weighted `BCEWithLogitsLoss` for label imbalance handling
- calibrated label-specific thresholds for operational use

This distinction between raw probabilities and calibrated thresholds is important. The dashboard does not simply use a default 0.5 threshold for every topic; instead, threshold selection is tuned to better balance precision, recall, and dashboard interpretability.

## Suggested environment

A Python environment with the following packages is recommended:

- `pandas`
- `numpy`
- `scikit-learn`
- `torch`
- `transformers`
- `datasets`
- `sentence-transformers`
- `bertopic`
- `umap-learn`
- `hdbscan`
- `matplotlib`
- `seaborn`
- `plotly`
- `openpyxl`
- `streamlit`

## How to use the project

A typical usage order is:

1. Run the review extraction and preparation notebook.
2. Create the manual labeling subset.
3. Label the selected reviews manually.
4. Run exploratory clustering for qualitative topic discovery.
5. Train and compare supervised classification models.
6. Run the final experiment notebook for model selection.
7. Apply the selected model and export dashboard-ready predictions.
8. Load the processed output into the Streamlit dashboard.

## Thesis context

This repository is part of a Master's thesis on NLP-based app review analytics. The focus is not only on predictive performance, but also on methodological transparency, interpretable topic structures, and practical usability in a dashboard setting.

The notebooks are therefore designed to document the full workflow step by step, including data preparation, exploratory analysis, model comparison, threshold calibration, and final deployment-oriented export.

## Notes

- Some file paths in the notebooks assume a local project folder structure under `../data/...`.
- Model and export artifacts are typically written to an `output/` directory.