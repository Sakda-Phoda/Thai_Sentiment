"""Phase 7 — Gradio demo for Thai Sentiment Classifier (WangchanBERTa).

Run locally:   python app/app.py
On HF Spaces: Set env var `MODEL_ID` to point to the Hugging Face Hub repo
              (not the local path), then place this file as app.py at the root of the Space.

Note: The Space requires its own requirements.txt (transformers, torch, gradio, sentencepiece).
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import torch
from transformers import pipeline

# local default: the fine-tuned model; on Spaces, override with env MODEL_ID = "<user>/<repo>"
_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_LOCAL = str(_REPO / "models" / "thai-sentiment" / "best")
MODEL_ID = os.environ.get("MODEL_ID", _DEFAULT_LOCAL)

device = 0 if torch.cuda.is_available() else -1
clf = pipeline("text-classification", model=MODEL_ID, tokenizer=MODEL_ID, device=device)


def predict(text: str) -> dict:
    if not text or not text.strip():
        return {}
    # top_k=None -> returns scores for all classes; always specify max_length
    scores = clf(text, top_k=None, truncation=True, max_length=256)
    if scores and isinstance(scores[0], list):  # some versions return nested lists
        scores = scores[0]
    return {item["label"]: float(item["score"]) for item in scores}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Thai Text", placeholder="Type a review or comment here..."),
    outputs=gr.Label(num_top_classes=4, label="Sentiment"),
    title="Thai Sentiment Classifier (WangchanBERTa)",
    description=(
        "Fine-tuned `airesearch/wangchanberta-base-att-spm-uncased` on Wisesight Sentiment Corpus "
        "(4 classes: positive / neutral / negative / question)"
    ),
    examples=[
        ["ร้านนี้อร่อยมากกก ประทับใจสุด ๆ"],
        ["บริการแย่มาก รอนานเกินไป"],
        ["เปิดกี่โมงคะ"],
        ["ของได้รับแล้วค่ะ แพ็คมาดีปกติ"],
    ],
)

if __name__ == "__main__":
    demo.launch()
