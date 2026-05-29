import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config_loader import get_app_config, resolve_project_path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@st.cache_resource
def load_model_bundle():
    """
    Loads the One-vs-Rest (OVR) binary models and their specific thresholds.
    Supports local execution and automatic Hugging Face Hub fallback.
    """
    cfg = get_app_config()
    
    export_dir = resolve_project_path(cfg.get("export_model_dir", "output/03_model_training/results_approach_loss_experiment/best_dashboard_model"))
    ovr_models_dir = export_dir / "ovr_models"
    thresholds_path = export_dir / "dashboard_thresholds.csv"

    HF_REPO_ID = "stadler93/health-insurance-models"

    active_labels = [
        'auth_registration',
        'tech_stability_crash',
        'general_feedback',
        'document_management',
        'smarthealth_epa_features'
    ]

    # 1. Load Thresholds
    threshold_map = {}
    if thresholds_path.exists():
        df_thresholds = pd.read_csv(thresholds_path)
        threshold_map = dict(zip(df_thresholds['label'], df_thresholds['threshold']))
    elif hf_hub_download is not None:
        try:
            local_csv_path = hf_hub_download(repo_id=HF_REPO_ID, filename="dashboard_thresholds.csv")
            df_thresholds = pd.read_csv(local_csv_path)
            threshold_map = dict(zip(df_thresholds['label'], df_thresholds['threshold']))
        except Exception:
            threshold_map = {lbl: 0.50 for lbl in active_labels}
    else:
        threshold_map = {lbl: 0.50 for lbl in active_labels}

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3. Load Tokenizer & Models
    models = {}
    tokenizer = None
    
    for label in active_labels:
        model_path = ovr_models_dir / label
        
        # Lokaler Ladeprozess (PC)
        if model_path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(str(model_path), local_files_only=True)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
        
        # Cloud-Fallback (Hugging Face Spaces) mit erzwungenem lokalen Caching gegen Timeouts
        elif hf_hub_download is not None:
            repo_subfolder = f"ovr_models/{label}"
            # local_files_only=False erlaubt das Herunterladen im Hintergrund
            model = AutoModelForSequenceClassification.from_pretrained(
                HF_REPO_ID, 
                subfolder=repo_subfolder,
                local_files_only=False
            )
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    HF_REPO_ID, 
                    subfolder=repo_subfolder,
                    local_files_only=False
                )
        else:
            raise FileNotFoundError(f"Keine Modelle unter {ovr_models_dir} oder HF-Hub-Anbindung gefunden.")

        model.to(device)
        model.eval()
        models[label] = model

    return tokenizer, models, threshold_map, active_labels, device


def predict_dashboard_labels(df, text_col="full_text"):
    """
    Runs the pipeline. The models are loaded lazily here if not already cached.
    """
    # Hier werden die Modelle erst geladen, wenn Inferenz benötigt wird!
    tokenizer, models, threshold_map, active_labels, device = load_model_bundle()

    texts = df[text_col].fillna("").astype(str).tolist()
    out = df.copy()

    if not texts:
        return out

    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    probs_dict = {}

    with torch.no_grad():
        for label, model in models.items():
            outputs = model(**enc)
            logits = outputs.logits.cpu().numpy()
            
            if logits.ndim > 1 and logits.shape[1] > 1:
                probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
            else:
                probs = sigmoid(logits).flatten()
                
            probs_dict[label] = probs

    predicted_topics_list = []

    for i in range(len(out)):
        labels_i = []
        for label in active_labels:
            p = float(probs_dict[label][i])
            thr = threshold_map.get(label, 0.50)
            pred = int(p >= thr)

            out.loc[out.index[i], f"prob_{label}"] = p
            out.loc[out.index[i], f"pred_{label}"] = pred

            if pred == 1:
                labels_i.append(label)

        predicted_topics_list.append("; ".join(labels_i))

    out["predicted_dashboard_labels"] = predicted_topics_list

    return out