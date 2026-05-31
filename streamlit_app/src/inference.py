import hashlib
import os

import pandas as pd
import streamlit as st

from config_loader import get_app_config, resolve_project_path

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None


HF_REPO_ID = "stadler93/health-insurance-models"

ACTIVE_LABELS = [
    "auth_registration",
    "tech_stability_crash",
    "general_feedback",
    "document_management",
    "smarthealth_epa_features",
]


def make_review_key(row, text_col="full_text"):
    """
    Create a stable hash key for a review row.

    The key combines the review text and selected metadata fields so that
    cached predictions can be reused across reruns for the same review.
    """
    raw = "||".join([
        str(row.get(text_col, "")),
        str(row.get("reviewdate", "")),
        str(row.get("rating", "")),
        str(row.get("sourcestore", "")),
        str(row.get("appname", "")),
        str(row.get("country", "")),
        str(row.get("language", "")),
    ])
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False, ttl="7d")
def load_model_bundle():
    """
    Load the tokenizer, OVR models, thresholds, and target device.

    Loading strategy:
    - Prefer local exported models from the configured project path
    - Fall back to the Hugging Face Hub if local files are unavailable
    - Fall back to default thresholds if no threshold file can be loaded

    Streamlit caches this function as a shared resource because model loading
    is expensive and the returned objects are reused across predictions.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # Limit CPU thread usage to avoid excessive resource consumption.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.set_num_interop_threads(1)

    cfg = get_app_config()

    # Resolve the configured export directory that contains the trained models and threshold file created during model training.
    export_dir = resolve_project_path(
        cfg.get(
            "export_model_dir",
            "output/03_model_training/results_approach_loss_experiment/best_dashboard_model",
        )
    )
    ovr_models_dir = export_dir / "ovr_models"
    thresholds_path = export_dir / "dashboard_thresholds.csv"

     # Use HF_HOME if available; otherwise fall back to the default cache path commonly used in hosted environments.
    hf_cache_dir = os.getenv("HF_HOME", "/data/.huggingface")

    # Load label-specific decision thresholds from the local export first.
    # If that fails, try downloading them from the Hugging Face Hub.
    # If both options fail, use a default threshold of 0.50 for all labels.
    if thresholds_path.exists():
        df_thresholds = pd.read_csv(thresholds_path)
        threshold_map = dict(zip(df_thresholds["label"], df_thresholds["threshold"]))
    elif hf_hub_download is not None:
        try:
            local_csv_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="dashboard_thresholds.csv",
                cache_dir=hf_cache_dir,
            )
            df_thresholds = pd.read_csv(local_csv_path)
            threshold_map = dict(zip(df_thresholds["label"], df_thresholds["threshold"]))
        except Exception:
            threshold_map = {lbl: 0.50 for lbl in ACTIVE_LABELS}
    else:
        threshold_map = {lbl: 0.50 for lbl in ACTIVE_LABELS}

    # Select GPU if available; otherwise run on CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = {}
    tokenizer = None

    # Load one binary classifier per dashboard label.
    # The setup assumes a one-vs-rest (OVR) architecture.
    for label in ACTIVE_LABELS:
        model_path = ovr_models_dir / label

        # Prefer local model artifacts if they exist.
        if model_path.exists():
            model = AutoModelForSequenceClassification.from_pretrained(
                str(model_path),
                local_files_only=True,
            )

            # Load the tokenizer only once.
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    str(model_path),
                    local_files_only=True,
                )

        # If no local model is available, fall back to the Hugging Face Hub.
        elif hf_hub_download is not None:
            repo_subfolder = f"ovr_models/{label}"

            model = AutoModelForSequenceClassification.from_pretrained(
                HF_REPO_ID,
                subfolder=repo_subfolder,
                local_files_only=False,
                cache_dir=hf_cache_dir,
            )

            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(
                    HF_REPO_ID,
                    subfolder=repo_subfolder,
                    local_files_only=False,
                    cache_dir=hf_cache_dir,
                )
        else:
            raise FileNotFoundError(
                f"Keine Modelle unter {ovr_models_dir} gefunden und kein HF-Hub-Fallback verfügbar."
            )

        # Move the model to the target device and switch to evaluation mode.
        model.to(device)
        model.eval()
        models[label] = model

    return tokenizer, models, threshold_map, ACTIVE_LABELS, device


@st.cache_data(show_spinner=False, ttl="7d", max_entries=200000)
def predict_single_review_cached(review_key, text, batch_size=16, max_length=128):
    """
    Predict dashboard labels for a single review and cache the result.

    The review_key is used as the cache identity, while the text is the actual
    model input. Returning a plain dictionary keeps the cached output compact
    and easy to merge back into a DataFrame.
    """
    import torch

    tokenizer, models, threshold_map, active_labels, device = load_model_bundle()

    # Convert missing values to an empty string so tokenization remains robust.
    text = "" if text is None else str(text)

    # Tokenize the review text once and reuse the encoded tensors for all OVR models.
    enc = tokenizer(
        [text],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    entry = {}
    labels_i = []

    # Use inference_mode for efficient prediction without gradient tracking.
    with torch.inference_mode():
        for label, model in models.items():
            logits = model(**enc).logits

            # Support both common classifier output formats:
            # - two logits for binary softmax classification
            # - one logit for sigmoid-based binary classification
            if logits.ndim > 1 and logits.shape[1] > 1:
                prob = float(torch.softmax(logits, dim=-1)[0, 1].cpu().item())
            else:
                prob = float(torch.sigmoid(logits.squeeze(-1))[0].cpu().item())

            thr = float(threshold_map.get(label, 0.50))
            pred = int(prob >= thr)

            entry[f"prob_{label}"] = prob
            entry[f"pred_{label}"] = pred

            if pred == 1:
                labels_i.append(label)

    # Store a compact semicolon-separated label summary for downstream display.
    entry["predicted_dashboard_labels"] = "; ".join(labels_i)
    return entry


def predict_dashboard_labels(df, text_col="full_text", batch_size=16, max_length=128):
    """
    Predict dashboard labels for all reviews in a DataFrame.

    The function creates a stable review key for each row, runs cached
    single-review inference, and appends the prediction columns to the input.
    """
    out = df.copy()

    # Return early if there is nothing to predict or the required text column is missing.
    if out.empty or text_col not in out.columns:
        return out

    # Ensure text input is always a valid string before hashing and inference.
    out[text_col] = out[text_col].fillna("").astype(str)
    
    # Create a review-specific cache key so repeated app runs can reuse predictions.
    out["review_key"] = out.apply(lambda row: make_review_key(row, text_col=text_col), axis=1)

    pred_rows = []
    for _, row in out.iterrows():
        pred_rows.append(
            predict_single_review_cached(
                review_key=row["review_key"],
                text=row[text_col],
                batch_size=batch_size,
                max_length=max_length,
            )
        )

    pred_df = pd.DataFrame(pred_rows)
    
    # Concatenate the original review data with the prediction output columns.
    out = pd.concat([out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)

    return out