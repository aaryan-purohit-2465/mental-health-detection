🧠 Mental Health Detection

A machine-learning based project that predicts whether a piece of text shows possible mental distress or is neutral.
This version includes a BERT-based classifier, an evaluation pipeline, and a full Streamlit UI for real-time predictions.

🚀 Features

Binary text classification (Neutral vs Possible Mental Distress)

BERT fine-tuned baseline model

Streamlit UI with confidence scores

Model evaluation: accuracy, F1-score, confusion matrix

Organized, modular repo structure

Local demo model checkpoint included

📁 Project Structure
mental-health-detection/
│
├── checkpoints/              # Saved model (demo_model/)
├── configs/                  # Config files
├── data/                     # Dataset placeholder
├── docs/                     # Screenshots, documentation
│   └── streamlit_ui.png
├── results/                  # Evaluation results
├── src/
│   ├── app/
│   │   └── streamlit_app.py  # Streamlit UI
│   ├── eval/
│   │   └── evaluate.py       # Metrics + confusion matrix
│   ├── models/
│   │   ├── train.py          # Training script
│   │   └── predict.py        # Inference script
│   └── __init__.py
│
├── requirements.txt
└── README.md

🛠️ Installation
1️⃣ Create a virtual environment
python -m venv venv

2️⃣ Activate it

Windows:

.\venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

▶️ Run the Streamlit UI

Run the demo interface:

streamlit run src/app/streamlit_app.py


Then open:

http://localhost:8501

🧪 Model Evaluation

You can check the performance using:

python src/eval/evaluate.py


This prints:

Accuracy

Precision

Recall

F1-score

Confusion Matrix

The evaluation is performed on a small sample dataset (demo purpose).


🎯 Purpose of This Project

This project was built to explore text classification, mental health detection, and deployment-ready ML workflows with:

Clean code

Modular structure

Real-time inference

Beginner-friendly design

📌 Version

v0.1 — Demo Release (2025-11-28)
Includes BERT baseline + Streamlit UI + evaluation.