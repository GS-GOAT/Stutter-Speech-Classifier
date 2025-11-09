# StutterFold: A Retrieval-Augmented Conformer for Multi-label Stuttering Detection

[](https://www.python.org/downloads/release/python-390/)
[](https://www.tensorflow.org/)
[](https://huggingface.co/)
[](https://opensource.org/licenses/MIT)

This repository contains the code for **StutterFold**, a novel, retrieval-augmented pipeline for the challenging task of multi-label stuttering event detection.

Instead of a simple classifier, this model learns to **classify by comparison.** It queries a specialized knowledge base (a Faiss index) to find acoustically and semantically similar clips, and then uses a **Cross-Attention** mechanism to reason over these "neighbors" before making a final prediction.

This pipeline achieves a **weighted average F1-score of 0.64** on a strict, speaker-independent split of the SEP-28k dataset, setting a new state-of-the-art for acoustic-only models and demonstrating remarkable, interpretable behavior.

-----

## 1\. Project Overview & Contribution

Stuttering is a complex speech disorder where disfluencies (e.g., blocks, prolongations, repetitions) often co-occur. This makes it a natural **multi-label classification problem**. This project addresses this challenge head-on, using a strict speaker-independent protocol to prevent data leakage and ensure real-world applicability.

### Key Contributions

1.  **Metric-Learned Retrieval Space**: A custom metric-learning stage trains a BiGRU embedder with **Triplet Loss**. This creates a *semantically-aware* retrieval space where audio clips are clustered by their multi-label Jaccard similarity, not just acoustic similarity.
2.  **Retrieval-Augmented Conformer (RAC)**: The final classifier, "StutterFold," uses a **Cross-Attention** module to fuse a query clip's acoustic features with a rich set of data from its $k$-nearest neighbors (their acoustics, ground-truth labels, and similarity scores).
3.  **State-of-the-Art Results**: Achieves a **0.64 weighted F1-score**, significantly outperforming prior acoustic-only benchmarks. This includes a massive 50-point F1-score improvement on the `Block` class (0.69 vs. 0.16).
4.  **Full Model Interpretability**: The model's decisions are highly interpretable. By visualizing the cross-attention weights, we can see *exactly which neighbors* the model "paid attention to" and *why* it made its decision, providing a "LIME-like" analysis built directly into the architecture.

-----

## 2\. The Problem: SEP-28k Dataset Analysis

The pipeline is trained on the SEP-28k dataset. After filtering out "NoStutter"-only clips, the task becomes classifying the 29,783 clips that contain at least one disfluency.

### 2.1. Class Imbalance (Original Filtered Data)

The dataset is highly imbalanced, which we address with instance-balanced augmentation in Stage 1.

| Class | Clips with Label (Original, Filtered) |
| :--- | :---: |
| Block | 8081 |
| Interjection | 5934 |
| Prolongation | 5629 |
| SoundRep | 3486 |
| WordRep | 2759 |

### 2.2. Label Cardinality (Why Multi-Label?)

A simple multi-class approach is insufficient. Over **56%** of all disfluent clips contain **two or more** co-occurring stuttering events.

| Labels per Clip | Count of Clips | Percentage |
| :---: | :---: | :---: |
| 1 | 12975 | 43.57% |
| 2 | 10964 | 36.81% |
| 3 | 4746 | 15.94% |
| 4 | 1014 | 3.40% |
| 5 | 84 | 0.28% |

-----

## 3\. Methodology: The 3-Stage StutterFold Pipeline

Our approach is a modular, 3-stage pipeline that transforms raw audio into highly accurate, interpretable predictions.

### Stage 1: Feature Extraction & Augmentation

  - **Input**: Raw audio from the SEP-28k dataset.
  - **Process**:
    1.  **Speaker-Aware Split**: Data is split into train/val/test sets, ensuring no speaker appears in more than one set.
    2.  **Instance-Balanced Augmentation**: The training set is "intelligently" oversampled. We use `audiomentations` to create new copies of clips, with the number of copies based on the rarity of the labels in each clip. This results in a balanced training set of \~10.5k samples per class.
    3.  **Feature Extraction**: All clips are passed through a frozen `facebook/wav2vec2-base-960h` model.
  - **Output**: Compressed `.npz` files containing feature arrays (`150x768`) and one-hot label vectors.

### Stage 2: Metric Learning & Indexing

  - **Input**: The augmented training features (`X_train`) and labels (`y_train`).
  - **Process**:
    1.  **Triplet Mining**: We create training triplets $(A, P, N)$ based on **Jaccard similarity** of the label vectors. This teaches the model to group clips by *label co-occurrence*.
    2.  **Metric Learning**: We train a "powerful embedder" (BiGRU + Attention Pooling) using Triplet Loss to learn a new embedding space.
    3.  **Indexing**: We use this trained embedder to compute new embeddings for all training clips and build a **Faiss** index for fast $k$-NN search.
  - **Output**: A set of weights (`powerful_triplet_embedder.h5`) and a Faiss index (`train_data_triplet.index`).

### Stage 3: Retrieval-Augmented Classification (RAC)

  - **Input**: A query clip and its pre-computed features.
  - **Process**:
    1.  **Retrieve**: The query's features are passed through the Stage 2 embedder to find its $k=10$ nearest neighbors from the Faiss index.
    2.  **Gather**: We fetch the neighbors' acoustic features (`150x768`), ground-truth labels, and similarity scores.
    3.  **Fuse**: Our final **StutterFold** model (a Conformer + Cross-Attention architecture) takes all 4 inputs (query clip, neighbor features, neighbor labels, neighbor similarities) and uses a cross-attention module to "reason" over the neighbors and produce a final, highly-accurate prediction.
  - **Output**: A multi-label prediction vector.

-----

## 4\. Repository Structure

The code is organized by the pipeline stages for clarity and reproducibility.

  - **`Stage 1`**: Wav2Vec2 extraction and augmentation.
  - **`Stage 2`**: Training the Triplet Loss embedder and building the Faiss index.
  - **`Stage 3`**: Training the final StutterFold classifier.
  - **`Stage 4`**: Contains images and plots.

(*code for the later stage in https://github.com/GS-GOAT/Stutter-Speech-Classifier/blob/main/stut-vecs/stut-vecs7.ipynb*)

-----

## 5\. Results & Model Interpretability

### 5.1. Final Classification Report

Our final model achieves a **0.64 weighted F1-score**. The model is characterized by exceptionally high recall (0.76), making it excellent at *finding* disfluencies, with a moderate precision (0.56) that can be tuned by adjusting the classification threshold.

| | precision | recall | f1-score | support |
|:---|---:|---:|---:|---:|
| Prolongation | 0.53 | 0.71 | 0.61 | 1335 |
| Block | 0.55 | 0.90 | 0.69 | 1982 |
| SoundRep | 0.41 | 0.70 | 0.52 | 886 |
| WordRep | 0.51 | 0.45 | 0.48 | 917 |
| Interjection | 0.69 | 0.84 | 0.76 | 1833 |
| **weighted avg** | **0.56** | **0.76** | **0.64** | **6953** |

### 5.2. Benchmark Comparison

The StutterFold pipeline dramatically outperforms the prior acoustic-only SOTA from Bayerl et al. (2023), especially on the most common `Block` class, which prior work almost completely failed to model.

| Disfluency Type | Bayerl et al. '23 F1 | **Our F1 (StutterFold)** | $\Delta$ |
| :--- | :---: | :---: | :---: |
| Block | 0.16 | **0.69** | **+0.53** |
| Interjection | 0.77 | **0.76** | -0.01 |
| Prolongation | 0.51 | **0.61** | **+0.10** |
| Sound Repetition | 0.50 | **0.52** | +0.02 |
| Word Repetition | 0.62 | **0.48** | -0.14 |
| **Weighted Avg F1** | **\~0.55** | **0.64** | **+0.09** |

-----
## 6\. Dependencies

  - Python 3.9+
  - `tensorflow` (2.16+)
  - `faiss-cpu` (or `faiss-gpu`)
  - `transformers`
  - `numpy`
  - `pandas`
  - `scikit-learn`
  - `audiomentations`
  - `matplotlib`, `seaborn`

-----

## 7\. Contact

For questions or collaboration, please open an issue in this repository.

---