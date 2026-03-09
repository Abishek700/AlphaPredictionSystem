# Alpha Recommendation System

This repository contains the implementation, report, and supporting materials for the **Alpha Recommendation System**, a machine learning-based framework for recommending context-dependent significance levels (α) in hypothesis testing.

The project explores how replication data and study characteristics can be used to move beyond the conventional fixed significance level of α = 0.05.

---

## Repository Structure

### Code
Contains the implementation of the Alpha Recommendation System.

This includes:
- the Streamlit application
- machine learning model
- dataset used for evaluation
- supporting source code

Instructions for running the system are provided inside the `Code` directory.

---

### MLbib
Contains literature used during the project.

---

### report
Contains the LaTeX source files for the thesis and report in pdf.

---

### author
Contains the details of author.

---

### Presentations
Contains presentation slides related to the project.

---

## Project Overview

The system uses a **Random Forest regression model** trained on replication study data to estimate a baseline significance level (α).

The predicted value is then adjusted using contextual factors such as:

- risk tolerance
- test direction
- multiple hypothesis testing

The final output is a recommended alpha value tailored to the characteristics of a study.

---

## Author

Abishek