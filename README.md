# Clustering African Countries Based on COVID-19 Spread Patterns

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/codermiki/clustering-african-countries-based-on-COVID-19-spread-patterns/blob/main/Clustering_African_Countries_Based_on_COVID_19_Spread_Patterns.ipynb)

## 1. Project Overview

This project implements **Unsupervised Machine Learning** techniques to cluster African countries based on their COVID-19 spread patterns. The primary objective is to group countries with similar pandemic trajectories to facilitate fair comparison and provide actionable insights for regional public health planning.

The core metric used for clustering is the **14-day cumulative COVID-19 cases per 100,000 population**. This normalized metric allows for an equitable comparison between countries with vastly different population sizes.

## 2. Objectives

-  **Identify Patterns**: Detect distinct patterns of COVID-19 transmission across the African continent.
-  **Comparative Analysis**: Group countries into clusters to enable comparative analysis of mitigation strategies and outcomes.
-  **Public Health Insights**: Provide data-driven insights to support policy formulation and resource allocation.

## 3. Dataset

The project utilizes data reported by the **European Centre for Disease Prevention and Control (ECDC)**.

-  **Content**: Daily records of COVID-19 cases, deaths, and population data for African countries.
-  **Preprocessing**: The raw data is aggregated and normalized to calculate the 14-day cumulative cases per 100,000 population for each country.

## 4. Methodology

### 4.1. Preprocessing

Data cleaning and feature scaling were performed using `StandardScaler` to ensure that all features contribute equally to the distance computations in clustering algorithms.

### 4.2. Algorithms

Three unsupervised machine learning algorithms were implemented and compared:

1. **K-Means Clustering**: Partitioning n observations into k clusters in which each observation belongs to the cluster with the nearest mean.
2. **Agglomerative Hierarchical Clustering**: Building a hierarchy of clusters using a bottom-up approach.
3. **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**: Efficiently clustering large datasets by building a valid Clustering Feature (CF) tree.

### 4.3. Evaluation

The **Silhouette Score** was used to evaluate the quality of the clusters. This metric measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation).

## 5. Key Results

### 5.1. Algorithm Performance

| Algorithm     | Silhouette Score |
| :------------ | :--------------- |
| **K-Means**   | **0.796**        |
| Agglomerative | 0.780            |
| BIRCH         | 0.778            |

**K-Means** achieved the highest silhouette score, indicating the most well-defined and distinct clusters.

### 5.2. Cluster Interpretation (K-Means)

The analysis identified three distinct clusters of countries:

-  **Cluster 0 (Moderate Impact)**: Countries experiencing intermittent waves of infection.
-  **Cluster 1 (Low Impact)**: Countries with consistently low transmission rates.
-  **Cluster 2 (High Impact)**: Countries with sustained high transmission rates (e.g., South Africa).

### 5.3. Visualization

A choropleth map of Africa was generated to visualize the spatial distribution of these clusters, revealing clear geographic disparities in the pandemic's impact.

## 6. Dependencies

The project uses Python and standard data science libraries. Key dependencies include:

-  `pandas`
-  `numpy`
-  `matplotlib`
-  `seaborn`
-  `scikit-learn`
-  `scipy`
-  `plotly`

## 7. Installation and Usage

### Option 1: Run on Google Colab (Recommended)

The easiest way to run this project is using Google Colab, which requires no local setup.

1. Click the **Open in Colab** badge at the top of this README.
2. Or click here: [Open Notebook in Google Colab](https://colab.research.google.com/github/codermiki/clustering-african-countries-based-on-COVID-19-spread-patterns/blob/main/Clustering_African_Countries_Based_on_COVID_19_Spread_Patterns.ipynb)
3. Once open, run all cells to execute the analysis.

### Option 2: Run Locally

1. **Clone the repository:**

   ```bash
   git clone https://github.com/codermiki/clustering-african-countries-based-on-COVID-19-spread-patterns.git
   cd clustering-african-countries-based-on-COVID-19-spread-patterns
   ```

2. **Install dependencies:**

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy plotly
   ```

3. **Run the analysis:**
   Open the Jupyter Notebook `Clustering_African_Countries_Based_on_COVID_19_Spread_Patterns.ipynb` in your preferred environment (Jupyter Lab, etc.) and execute the cells.

## 8. Acknowledgments

-  Data provided by the **European Centre for Disease Prevention and Control (ECDC)**.
