````markdown
# Prototype Proposal: One‐Dimensional Semantic Spectrum via PCA Reduction

**Repository Path**: `IP/proposals/semantic_pca_spectrum.md`  
**Author**: [Your Name]  
**Date**: 2025-06-12

---

## 1. Executive Summary

This document proposes the design and implementation of a prototype tool that reduces high-dimensional semantic embeddings down to a single axis via Principal Component Analysis (PCA). The resulting 1D “spectrum” can be used to position objects, terms, or documents along a continuum of meaning—enabling quick comparisons, clustering, anomaly detection, or UI visualizations.  

Key benefits:

- **Interpretability**: A single scalar per item aligns with human notions of a “left–right” or “low–high” semantic scale.  
- **Simplicity**: PCA1 can be computed in milliseconds for up to millions of vectors using optimized libraries.  
- **Integrability**: Exposes a REST/gRPC API and/or Python library for downstream systems (e.g., search UIs, dashboards, recommendation engines).  

---

## 2. Background & Motivation

Modern NLP and multimodal systems embed text, images, and structured data into high-dimensional vector spaces (e.g., BERT, CLIP). While these embeddings excel at capturing nuance, their dimensionality makes direct human consumption challenging:

1. **Visualization**: Plotting >2-D requires dimensionality reduction (t-SNE, UMAP, PCA) but most UIs cannot handle >2 axes.  
2. **Ranking & Filtering**: Scalar scores are more intuitive for ordering, ordinal binning, or thresholding.  
3. **Resource Constraints**: Real-time dashboards need simple computations.

By extracting the first principal component (PCA1), we capture the direction of maximal variance—which often corresponds to the dominant semantic axis in a given dataset—and render it as a single float. This prototype will demonstrate feasibility, performance, and UX integration.

---

## 3. Objectives

- **O1**: Ingest a corpus of items with precomputed embeddings (or compute embeddings on‐the‐fly).  
- **O2**: Compute PCA1 across the corpus and project each item onto this axis.  
- **O3**: Expose results via a programmatic interface (Python library & RESTful service).  
- **O4**: Provide a minimal web-based visualization (horizontal spectrum) for demonstration.  
- **O5**: Evaluate explainability (variance captured) and performance (throughput, latency).  

---

## 4. Scope

- **In‐Scope**  
  - Single axis (1D) PCA reduction of arbitrary embedding dimensions (≥2).  
  - Support for loading embeddings in NumPy, PyTorch, or TensorFlow formats.  
  - Basic Python CLI and library interface (`compute_spectrum.py`).  
  - Dockerized REST API for remote queries.  
  - Web demo using D3.js or plain HTML+SVG.  
  - Unit tests and integration tests.

- **Out‐of‐Scope** (for prototype)  
  - Multi‐dimensional interactive UIs (beyond 1D).  
  - Dynamic incremental PCA updates (dataset assumed static per run).  
  - Advanced manifold‐learning methods (UMAP, t-SNE).  
  - Production hardening, authentication, RBAC.

---

## 5. Technical Approach

### 5.1 Data Ingestion

- **Formats**:  
  - `.npy` arrays of shape `(N, D)`  
  - PyTorch `torch.Tensor` on CPU  
  - JSON/CSV of lists  

- **Interface**:  
  ```python
  from semantic_spectrum import EmbeddingLoader

  loader = EmbeddingLoader(source="embeddings.npy")  
  X: np.ndarray = loader.load()  # shape (N, D)
  item_ids: List[str] = loader.item_ids()  # optional mapping
````

### 5.2 Preprocessing

* **Centering**: Subtract mean vector (`X_centered = X - X.mean(axis=0)`).
* **Optional Scaling**: Standardize features if embeddings have uneven scales (via `StandardScaler`).

### 5.3 PCA Computation

* **Library**: `scikit-learn`’s `PCA(n_components=1)` or `sklearn.utils.extmath.randomized_svd` for large D.
* **Computation**:

  ```python
  from sklearn.decomposition import PCA

  pca = PCA(n_components=1)
  coords_1d = pca.fit_transform(X_centered).ravel()  # shape (N,)
  explained_variance = pca.explained_variance_ratio_[0]
  principal_axis = pca.components_[0]  # for interpretability
  ```

### 5.4 API & Library Design

* **Python Package**:

  * `semantic_spectrum.compute()` → returns `(item_ids, coords_1d, explained_variance)`.
  * `semantic_spectrum.plot_spectrum()` → saves/returns a matplotlib figure.

* **REST Service** (FastAPI + Uvicorn + Docker):

  * **POST** `/compute`

    * Payload: embeddings file (URL or multipart upload)
    * Response: JSON with `{"ids": [...], "coords": [...], "variance": 0.72}`
  * **GET** `/spectrum.png`

    * Query params: `?top=50&bottom=50`
    * Response: Spectrum image.

### 5.5 Visualization

* **Web Demo**:

  * Simple HTML/JS page loading JSON from `/compute`, rendering an SVG line with labeled ticks.
  * Use minimal dependencies (D3.js optional).

```html
<div id="spectrum"></div>
<script>
fetch('/compute')
  .then(r => r.json())
  .then(data => {
    // map coords to pixel positions, append <circle> + <text> in SVG
  });
</script>
```

---

## 6. Architecture Diagram

```
+----------------+        +----------------+        +----------------+
|                |   1    |                |   2    |                |
|   Client/UI    +------->+   REST API     +------->+  PCA Engine    |
| (Browser/CLI)  |        |  (FastAPI)     |        | (scikit-learn) |
+----------------+        +---------------++        +----------------+
                                  | 3
                                  v
                             +----------+
                             |  Data    |
                             | Storage  |
                             | (Npy/DB) |
                             +----------+
```

1. Client sends embeddings or requests existing dataset.
2. API forwards to PCA engine.
3. Coordinates returned and optionally cached.

---

## 7. Implementation Plan

| Phase | Tasks                                                                      | Deliverables                                | Duration |
| ----- | -------------------------------------------------------------------------- | ------------------------------------------- | -------- |
| 1     | Project scaffolding, CLI stub, Dockerfile, dependencies                    | `pyproject.toml`, `Dockerfile`, `README.md` | 1 week   |
| 2     | Embedding loader abstraction, preprocessing module                         | `semantic_spectrum.loader.py`               | 1 week   |
| 3     | PCA computation module, unit tests for correctness                         | `semantic_spectrum.pca.py`, test suite      | 1 week   |
| 4     | Python CLI & library API, packaging distribution                           | `compute_spectrum.py`, `pip install .`      | 1 week   |
| 5     | FastAPI service with endpoints, Docker containerization, integration tests | `app.py`, `docker-compose.yml`              | 1 week   |
| 6     | Web demo page, documentation, end-to-end tests                             | `docs/`, `demo/index.html`                  | 1 week   |
| 7     | Performance benchmarking, variance analysis report, final cleanup          | Benchmark scripts, `docs/performance.md`    | 2 weeks  |

**Total Duration**: \~8 weeks

---

## 8. Evaluation & Success Metrics

1. **Explained Variance**

   * Target: PCA1 captures ≥40% of total variance in typical datasets.
2. **Accuracy of Ordering**

   * Qualitative validation by domain experts to ensure semantic ordering makes sense.
3. **Performance**

   * Throughput ≥10,000 vectors/sec for D ≤ 768 on commodity CPU.
   * API latency ≤200 ms per request on moderate hardware.
4. **Usability**

   * Clear CLI UX, comprehensive error messages.
   * Intuitive web demo with labels and tooltips.

---

## 9. Risks & Mitigations

| Risk                                          | Probability | Impact | Mitigation                                                              |
| --------------------------------------------- | ----------- | ------ | ----------------------------------------------------------------------- |
| PCA1 fails to align with human intuition      | Medium      | Medium | Collect user feedback; consider orthogonal rotation or supervised axis. |
| Large D or N causes memory/time blowup        | High        | High   | Use incremental or randomized SVD; batch streaming.                     |
| Dependency conflicts (scikit-learn, FastAPI…) | Low         | Low    | Pin versions; isolate in Docker.                                        |
| Web demo performance on large N               | Medium      | Low    | Paginate or sample top/bottom K items.                                  |

---

## 10. Resource & Dependency List

* **Languages & Frameworks**

  * Python ≥3.10, FastAPI, scikit-learn, NumPy, Uvicorn, Docker
* **Libraries for Demo**

  * D3.js (optional), plain HTML/CSS/JS
* **Infrastructure**

  * CI/CD (GitHub Actions), Docker Hub, simple object storage (S3/GCS)
* **Team**

  * 1× Backend Engineer (PCA & API)
  * 1× Frontend Engineer (Web Demo)
  * 0.5× Data Scientist (Validation & Metrics)

---

## 11. Next Steps

1. Review & approve proposal in `IP/proposals/`.
2. Spin up project scaffold via `cookiecutter` or manual.
3. Kick off Phase 1 sprint; assign tasks and set up CI.
4. Schedule demo checkpoint at end of Phase 3 (PCA engine ready).

---

## Appendix A: Sample Python Snippet

```python
import numpy as np
from sklearn.decomposition import PCA
from semantic_spectrum.loader import EmbeddingLoader

# 1. Load embeddings
loader = EmbeddingLoader("data/embeddings.npy", ids_file="data/ids.txt")
X = loader.load()           # shape (N, D)
ids = loader.item_ids()

# 2. Center & optional scale
Xc = X - X.mean(axis=0)

# 3. Compute PCA1
pca = PCA(n_components=1)
coords = pca.fit_transform(Xc).ravel()
print("Explained variance:", pca.explained_variance_ratio_[0])

# 4. Output spectrum
for item_id, coord in zip(ids, coords):
    print(f"{coord:.4f}\t{item_id}")
```

---

*End of Proposal*

```
```
