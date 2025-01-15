# LegalDocumentRetrieval
Team Akasuki - SoICT Hackathon 2024 

# Team Akasuki - SoICT Hackathon 2024 Solution: Legal Document Retrieval

[![Status](https://img.shields.io/badge/Status-In%20Progress-green)](https://shields.io/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

## ✨ **Design Idea**

**🎯 Task:** Retrieve relevant legal documents related to a query from a given corpus.

**🤝 Team:** Akatsuki

### 🚀 **Solution Approach**

Combine the power of two open-source Vietnamese language models:

1. **`bkai-foundation-models/vietnamese-bi-encoder` (Bi-encoder)**
    ![Bi-encoder](https://img.icons8.com/color/48/000000/transformer.png)
2. **`namdp-ptit/ViRanker` (Cross-encoder)**
    ![Cross-encoder](https://img.icons8.com/color/48/000000/data-configuration.png)

### 🛠️ **Part 1: Fine-tuning the Bi-encoder**

*   **📝 Preprocessing and Embedding:**
    *   Utilize the `bi-encoder` to generate vector representations for queries and documents in the `corpus` and `training data`.
    ![Embedding](https://img.icons8.com/color/48/000000/word-embedding.png)

*   **🎯 Fine-tuning:**
    *   Construct contrasting `(query, document)` pairs:
        *   **✅ Positive pairs:** Obtained from the `training data`.
        *   **❌ Negative pairs:** Selected from pairs not present in the `training data` and exhibiting low `cosine` similarity.
            ![Positive-Negative](https://img.icons8.com/color/48/000000/positive-dynamic.png)
    *   Train the `bi-encoder` using `MultipleNegativesRankingLoss`.
        ![Training](https://img.icons8.com/color/48/000000/training.png)

### 🔍 **Part 2: Prediction with Public Test**

*   **📊 Query Embedding:**
    *   Employ the fine-tuned `bi-encoder` to embed queries from the `public test` set.
        ![Embedding](https://img.icons8.com/color/48/000000/word-embedding.png)

*   **🧲 Candidate Retrieval:**
    *   Calculate `cosine similarity` between query vectors and document vectors in the `corpus`.
    *   Select the top 50 most promising candidates.
        ![Cosine Similarity](https://img.icons8.com/color/48/000000/similarity.png)

*   **🥇 Re-ranking:**
    *   Utilize the `cross-encoder` (`namdp-ptit/ViRanker`) to re-evaluate and rank the 50 candidates.
        ![Ranking](https://img.icons8.com/color/48/000000/rank.png)

*   **🏆 Results:**
    *   Choose the top 10 candidates with the highest scores from the `re-ranking` step.
        ![Top 10](https://img.icons8.com/color/48/000000/top-badge.png)

---
