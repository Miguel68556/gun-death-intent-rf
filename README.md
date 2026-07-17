# gun-death-intent-rf
Math 457 Final Project - predicting gun death intent with random forest classifiers 
# Predicting Gun Death Intent with Random Forests

Team: Matthew Reardon, Miguel Gonzalez, Ryan Smith

---

## Project Overview

Using CDC firearm-deaths data from FiveThirtyEight (~100,798 records), we built Random Forest classifiers to predict the **intent** behind a gun death — suicide, homicide, accidental, or undetermined — from demographic and situational features (age, race, sex, education, place, police involvement, year, month).

## Data

- **Source:** [FiveThirtyEight `guns-data` (CDC-derived)](https://raw.githubusercontent.com/fivethirtyeight/guns-data/refs/heads/master/full_data.csv)
- **Size:** 100,798 records, 10 usable features after dropping the unnamed index column and an ambiguous `hispanic` column
- **Class imbalance:** Suicide and homicide dominate; accidental (~1.6%) and undetermined (~0.8%) are severely underrepresented

## Approach

1. **Exploratory data analysis** across sex, age, race, education, and place, focused on how each feature relates to intent
2. **Multiclass Random Forest** predicting all four intents
   - 75/25 stratified train/test split
   - Max-depth tuning from 1 to 20
   - 5-fold cross-validation
3. **Binary Random Forest** predicting Homicide vs. Suicide only, chosen as the final model after the multiclass model struggled with the minority classes

## Results

| Model | Test Accuracy | Macro Avg | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| Multiclass (4 intents), max depth 12 | 83% | 55% | 46% (macro) | 76% |
| **Binary (Homicide vs Suicide), max depth 11** | **85%** | **85%** | **0.88** | **0.92** |

5-fold CV on the binary model returned a mean score of 85% with a standard deviation of 0.7%, indicating stable generalization.

**Most important predictors:** race (white and black), age, and place of death (home vs. street) — consistent with what the EDA suggested.

## Limitations & Ethical Framing

These patterns reflect **broader social and structural conditions**, not individual characteristics. Racial differences in outcomes likely reflect systemic factors such as exposure to violence and economic opportunity, and location differences reflect well-known social risk patterns rather than anything intrinsic to a place. The dataset also has limitations: reporting biases, demographic labels that don't capture full identity, and possible misclassification.

The model is useful for identifying **population-level trends** and shouldn't be used to make assumptions about individuals — doing so could reinforce inequity rather than address it.

## Files

- ./Math_457_Final_Project.ipynb — full analysis notebook (also runnable directly in Google Colab via the badge at the top)
- `MATH457_Final_Report.pdf` — final written report

## My Contribution

I contributed to the exploratory data analysis, model interpretation, and the limitations and ethical-framing sections of the write-up, and served as the group's primary code operator during model development. I understand random forests conceptually — max-depth tuning, cross-validation, PR-AUC vs ROC-AUC, and why we moved from the multiclass to the binary model — but I do not yet write production Python from scratch. This project represents a real applied ML workflow that I can walk through, including what I'd do differently.
