# MLOPS-Experiment02 - Anime Recommendation System

A complete **MLOps pipeline** for building an Anime Recommendation System using Neural Collaborative Filtering.

## Project Overview

This project demonstrates a production-grade MLOps workflow for training and deploying a deep learning-based anime recommender system.

### Tech Stack

- **Python 3.10+**
- **TensorFlow / Keras**
- **Pandas, NumPy, Scikit-learn**
- **Google Cloud Storage** (Data Ingestion)
- **DVC** (Data Version Control)
- **Comet ML** (Experiment Tracking)
- **Opik** (LLM Evaluation - upcoming)
- **Git + GitHub**

## Project Structure

```bash
MLProject2/
├── src/                    # Source code
│   ├── data_processing.py
│   ├── model_training.py
│   ├── base_model.py
│   ├── main_pipeline.py
│   └── ...
├── config/                 # Configuration files
├── artifacts/              # Model, weights, processed data
├── logs/                   # Application logs
├── utils/                  # Common utilities
├── pipeline/               # Pipeline scripts
├── .dvc/                   # DVC configuration
├── requirements.txt
├── setup.py
├── pyproject.toml
└── README.md
