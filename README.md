# Yelp Capstone Project

- **Project Title:** Predicting Business Success and Closure Using Yelp Reviews and Metadata
- **Team:** Kayode Babalola, Qianru Deng (Charlotte), Damola Emmanuel Olayemi, Wangcheng Peng, Chinenye Alice Anieke
- **Date:** December 2025
- **Program:** St. Clair College – Data Analytics Capstone


## Repository Structure
```
/yelp-capstone/
├── data/
│   ├── raw/                ← Raw downloaded data
│   ├── processed/          ← Cleaned / transformed data   
├── code/                 ← Data extraction, cleaning, and loading scripts 
│   ├── data_overview
│   ├── data_cleaning
│   ├── data_eda
│   └── data_engineering
├── models/                 ← Machine learning models
├── reports/                ← Final reports / presentations / visualizations  
├── docs/                   ← Project documentation
└── README.md
```


## Overview

Analyze historical Yelp data to identify factors distinguishing successful businesses from those that close. Build predictive models to assess business risk and provide actionable insights via a Tableau dashboard.


## Objectives

* Clean and preprocess Yelp datasets.
* Perform EDA to uncover business and customer patterns.
* Build models:

  * K-Means clustering (customer/business segmentation)
  * Recommendation system
  * Sentiment analysis
* Visualize insights via Tableau dashboard.
* Provide actionable business recommendations.


## Scope

**In-Scope:** Data cleaning, EDA, feature engineering, model development, dashboard creation, reporting.
**Out-of-Scope:** Real-time deployment, paid data enrichment, third-party proprietary data.


## Deliverables

| Deliverable               | Due        |
| ------------------------- | ---------- |
| Project Charter           | Week 1-5   |
| Data Preparation Notebook | Week 6-7   |
| EDA & Feature Engineering | Week 8-9   |
| Model Building            | Week 10-11 |
| Tableau Dashboard         | Week 11-12 |
| Presentation              | Week 12    |
| Final Report              | Week 14    |


## Team Roles

| Name            | Role                       | Key Responsibility                          |
| --------------- | -------------------------- | ------------------------------------------- |
| Kayode Babalola | Project Lead               | Data cleaning, modeling, sentiment analysis |
| Charlotte Deng  | Recommendation & Dashboard | Recommendation system, Tableau dashboard    |
| Damola Emmanuel | Customer Segmentation      | K-Means, customer insights                  |
| Wangcheng Peng  | EDA & Documentation        | Business overview, feature engineering      |
| Chinenye Alice  | Feature & Marketing        | Feature creation, marketing strategy        |
| All Members     | Dashboard Collaboration    | Design & validation of dashboard            |


## Tools & Technologies

* **Data:** Yelp Open Dataset (JSON)
* **Programming:** Python (Pandas, NumPy, Scikit-learn, NLTK, TensorFlow)
* **Storage:** CSV
* **Visualization:** Tableau, Matplotlib, Seaborn
* **Project Management:** Jira, GitHub, Google Drive
* **Environment:** Google Colab, VSCode


## Timeline

| Phase                     | Weeks | Key Tasks                                    |
| ------------------------- | ----- | -------------------------------------------- |
| Planning & Setup          | 1–5   | Goals, dataset collection, roles             |
| Data Preparation          | 6–7   | Cleaning, preprocessing, feature engineering |
| EDA & Feature Engineering | 8–9   | Insights, features for models                |
| Model Development         | 8–9   | Clustering, recommendation, sentiment models |
| Dashboard                 | 10–11 | Tableau visualization                        |
| Presentation              | 12    | Slides & demo                                |
| Final Report              | 14    | Submission                                   |


## Risks

* Incomplete/corrupt data → Validate JSON & handle missing values
* Low model accuracy → Hyperparameter tuning & multiple algorithms
* Time constraints → Agile sprints & Jira tracking
* Limited computing resources → Google Colab Pro, optimize code
* GitHub sync issues → Regular commits & backups


## Communication

* Weekly progress reports
* Jira updates
* Bi-weekly supervisor meetings
* Final presentation


## Success Criteria

* Cleaned and validated Yelp datasets
* Models achieve meaningful segmentation, recommendations, and sentiment classification
* Tableau dashboard communicates actionable insights clearly
* Peer-reviewed deliverables
* Positive feedback from faculty/stakeholders


## Post-Project Evaluation

* Retrospective: lessons learned
* Document reproducible workflow
* Share dashboard/results online
* Reflect on skills developed (EDA, ML, visualization)


## Appendix

* **Dataset:** [Yelp Open Dataset](https://www.yelp.com/dataset)
* **Google Drive:** [Project Files](https://drive.google.com/drive/folders/1QMCqYJM7ArOEBiukt-KppC2XYW74PUVX?usp=sharing)

