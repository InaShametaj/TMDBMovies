# TMDBMovies

# 🎬 Movie Clustering with KMeans & DBSCAN

Unsupervised clustering analysis of the [TMDB 5000 Movies dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) using **KMeans** and **DBSCAN**, with full cluster evaluation via the **Elbow method**, **SSE**, and **Silhouette score**.

---

## 📁 Project Structure

```
├── tmdb_5000_movies.csv       # Dataset (download separately)
├── kmeans_jupyterlab.py       # KMeans clustering + visualizations
├── dbscan_jupyterlab.py       # DBSCAN clustering + visualizations
├── dbscan_silhouette.py       # DBSCAN silhouette evaluation + eps sweep
├── kmeans_evaluation.py       # KMeans evaluation: Elbow / SSE / Silhouette
└── README.md
```

---

## 📊 Dataset

**TMDB 5000 Movies** — 4,803 movies with metadata scraped from The Movie Database.

| Feature | Type | Description |
|---|---|---|
| `budget` | int | Production budget (USD) |
| `revenue` | int | Box office revenue (USD) |
| `popularity` | float | TMDB popularity score |
| `runtime` | int | Film duration (minutes) |
| `vote_average` | float | Average user rating (0–10) |
| `vote_count` | int | Number of user votes |

> Rows with any zero value in these features are dropped before clustering (proxy for missing data), leaving **~3,227 clean rows**.

---

## ⚙️ Setup

### Requirements

```bash
pip install pandas numpy matplotlib scikit-learn
```

### Dataset

Download `tmdb_5000_movies.csv` from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place it in the **same folder** as the scripts.

### Running in JupyterLab

All scripts are written for JupyterLab. Paste the content of any script into a notebook cell and run it — no extra configuration needed.

---

## 🔵 KMeans Clustering — `kmeans_jupyterlab.py`

Groups movies into **k clusters** based on the 6 numeric features above.

**Pipeline:**
1. Clean & standardize features with `StandardScaler`
2. Run KMeans for k = 2–10 to find the elbow
3. Fit final model at optimal k
4. Visualize clusters in 2D via PCA projection
5. Profile each cluster with a heatmap and boxplots
6. Print top-5 movies per cluster by popularity

**Key results (k = 4):**

| Cluster | Size | Profile |
|---|---|---|
| 0 | 1,249 | Critically acclaimed, modest scale |
| 1 | 1,440 | Low-engagement mainstream |
| 2 | 98 | Blockbusters (high budget & revenue) |
| 3 | 440 | High-engagement mid-tier |

**Plots produced:**

| Plot | Description |
|---|---|
| Elbow + Silhouette | Side-by-side curves for selecting k |
| PCA scatter | 2D cluster projection with centroids |
| Feature heatmap | Normalised mean feature values per cluster |
| Boxplots | Feature distributions per cluster (log scale where needed) |

---

## 🟠 DBSCAN Clustering — `dbscan_jupyterlab.py`

Density-based clustering that discovers clusters of arbitrary shape and automatically flags outliers as **noise**.

**Pipeline:**
1. Clean & standardize features
2. Plot the **k-distance graph** to pick `eps`
3. Fit DBSCAN (`eps=1.5`, `min_samples=5`)
4. Visualize clusters + noise in PCA space
5. Profile clusters with heatmap and boxplots (noise excluded)
6. Print top-5 movies per cluster and from noise

**Key results (`eps=1.5`, `min_samples=5`):**

| Label | Size | Description |
|---|---|---|
| Cluster 0 | 3,127 | Dense mainstream core |
| Cluster 1 | 7 | Mega-budget outliers |
| Cluster 2 | 5 | Extreme outliers |
| Noise | 88 (2.7%) | No dense neighborhood |

**Tuning parameters:**

```python
EPS         = 1.5   # ↑ merges clusters  ↓ creates more noise
MIN_SAMPLES = 5     # ↑ stricter core points, more noise
```

> Use the k-distance plot to find the elbow — that value is the recommended `eps`.

---

## 📐 Cluster Evaluation

### KMeans — `kmeans_evaluation.py`

Evaluates KMeans across k = 2–10 using three complementary metrics.

| Metric | What it measures |
|---|---|
| **SSE (inertia)** | Sum of squared distances from each point to its centroid. Lower = tighter clusters. Look for the *elbow* where gains flatten. |
| **SSE drop %** | Percentage improvement in SSE going from k to k+1. Large drops signal meaningful new structure. |
| **Silhouette score** | How similar a point is to its own cluster vs. others. Range: −1 to +1. Higher = better separation. |

**Results on this dataset:**

| k | SSE | SSE drop % | Silhouette |
|---|---|---|---|
| 2 | 13,168 | — | **0.5227 ✓** |
| 3 | 10,803 | 18.0% | 0.2533 |
| 4 | 9,433 | 12.7% | 0.2439 |
| 5–10 | ↓ | ~8–12% | ~0.21–0.25 |

> Both the elbow (sharpest drop at k=2→3) and silhouette (peaks at k=2) agree: **k=2 is the statistically optimal split**.

**Plots produced:**

| Plot | Description |
|---|---|
| Elbow + Silhouette | Annotated side-by-side curves, red line at best k |
| Silhouette plot | Per-sample bands per cluster, red dashed = overall mean |

**Tables produced:**

| Table | Description |
|---|---|
| Evaluation table | SSE, SSE drop, SSE drop %, Silhouette — highlighted best values |
| Per-cluster summary | Mean / min / max silhouette, % of samples below overall mean |

---

### DBSCAN — `dbscan_silhouette.py`

Evaluates DBSCAN clusters with the silhouette score and sweeps `eps` to find the optimal value.

> **Note:** noise points (label = −1) are excluded from all silhouette calculations — they have no cluster assignment.

**Results (`eps=1.5`, `min_samples=5`):**

| Cluster | Size | Mean silhouette |
|---|---|---|
| 0 | 3,127 | 0.5434 |
| 1 | 7 | 0.4816 |
| 2 | 5 | 0.4290 |
| **Overall** | **3,139** | **0.5431** |

**eps sweep result:**

| eps | Clusters | Noise | Silhouette |
|---|---|---|---|
| 0.50 | 7 | 972 | 0.039 |
| 1.25 | 3 | 165 | 0.524 |
| **1.50** | **3** | **88** | **0.543 ✓** |
| 1.75+ | 1 | — | N/A |

**Plots produced:**

| Plot | Description |
|---|---|
| Silhouette plot | Per-sample bands per cluster, red dashed = overall mean |
| eps sweep chart | Silhouette score (line) + noise count (bars) vs eps |

---

## 🔍 KMeans vs DBSCAN — Quick Comparison

| | KMeans | DBSCAN |
|---|---|---|
| Cluster shape | Assumes spherical | Any shape |
| Outlier handling | Assigns everything | Labels outliers as noise |
| Requires k upfront | ✅ Yes | ❌ No |
| Sensitive to scale | ✅ Yes (use StandardScaler) | ✅ Yes |
| Best silhouette (this dataset) | 0.5227 (k=2) | 0.5431 (eps=1.5) |
| Clusters found | User-defined | 3 (auto) |

---

## 📌 Notes

- All features are standardized with `StandardScaler` before clustering — this is essential since features like `budget` (millions) and `vote_average` (0–10) are on very different scales.
- PCA is used **only for 2D visualization**, not for clustering itself.
- Silhouette scores below 0.2 suggest overlapping or poorly separated clusters.
