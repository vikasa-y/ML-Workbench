# ML Day 1 — Fundamentals

## Core Concepts

- **ML:** Learn patterns from data to make predictions.
- **X:** Input features
- **y:** Actual target
- **ŷ:** Model prediction
- **Sample:** One row
- **Feature:** One input column

## Learning Types

- **Supervised:** `X + y`
- **Unsupervised:** `X` only

## Prediction

Linear model:
ŷ = wx + b
w → weight
b → bias
Error & MSE
Error = y - ŷ
Squared Error = (y - ŷ)²
MSE = average of squared errors

Example:

y  = [2, 4, 6, 8, 10]
ŷ  = [4, 7, 10, 13, 16]

MSE = 18
Other Concepts
Regression → predicts numbers
Classification → predicts categories
Training → model learns
Testing → evaluate on unseen data
Overfitting → good training, poor unseen performance
Underfitting → model too simple
Data Leakage → unwanted information enters training
ML Workflow
Data → Clean → X/y → Train/Test
→ Train Model → Predict → Evaluate
→ Improve → Final Model


## 1. What is Machine Learning?

Machine Learning (ML) allows computers to learn patterns from data and use them to make predictions or decisions.

Traditional programming: Rules + Data → Output
ML: Data + Answers → Model → Predictions

## 2. Core Terms
Term	Meaning
X	Input features
y	Actual target / correct answer
ŷ	Model prediction
Sample	One row / observation
Feature	One input column
Model	Learned mathematical relationship/pattern
Parameter	Learned by the model
Hyperparameter	Set by us to control the model

Example:

X = [[1], [2], [3], [4], [5]]
y = [2, 4, 6, 8, 10]

X.shape = (5, 1) → 5 samples, 1 feature

## 3. Regression vs Classification
### a).Regression → predicts a number
Example: price, salary, temperature
### b). Classification → predicts a category
Example: spam/not spam, pass/fail

## 4. Supervised vs Unsupervised
### Supervised Learning

Has both:X + known y

The model learns to predict y.

### Unsupervised Learning
Has: X only

The model searches for patterns or structure.

## 5. Training vs Testing
Training data → model learns from it.
Test data → unseen data used to evaluate the model.

Goal: good generalization, not just memorizing training data.

## 6. Prediction

Simple linear model:

ŷ = wx + b
w → weight
b → bias
ŷ → prediction

## 7. Error and MSE
Error
Error = y - ŷ
Squared Error
Squared Error = (y - ŷ)²
Mean Squared Error
MSE = average of all squared errors

Formula:

MSE = (1/n) × Σ(y - ŷ)²

Lower MSE means lower squared prediction error on the evaluated data.

Example
Actual y       = [2, 4, 6, 8, 10]
Prediction ŷ   = [4, 7, 10, 13, 16]

Error          = [-2, -3, -4, -5, -6]
Squared error  = [4, 9, 16, 25, 36]

MSE = (4 + 9 + 16 + 25 + 36) / 5
    = 18
## 8. Why MSE?

Different models can make different predictions.

MSE converts their individual errors into one number, allowing us to compare model error.
```
Parameters
    ↓
Predictions (ŷ)
    ↓
Compare with y
    ↓
Error
    ↓
MSE
    ↓
Optimization
    ↓
Better parameters
```
## 9. Overfitting vs Underfitting
### Overfitting

Model performs very well on training data but poorly on unseen data.

### Underfitting

Model is too simple and performs poorly even on training data.

Generalization

Model performs well on new, unseen data.

## 10. Data Leakage

Data leakage happens when information that should not be available during prediction accidentally enters the training process.

Example:

Fitting preprocessing using the entire dataset before splitting into train/test data.

Rule:

Keep test information out of training.

## 11. Basic ML Workflow
```
Dataset
   ↓
Understand Data
   ↓
Clean / Preprocess
   ↓
Define X and y
   ↓
Train / Test Split
   ↓
Choose Model
   ↓
Train
   ↓
Predict (ŷ)
   ↓
Evaluate
   ↓
Error Analysis
   ↓
Improve / Tune
   ↓
Final Evaluation
```

## 12. Quick Revision
Remember these:

X  → Inputs
y  → Actual answer
ŷ  → Prediction

y - ŷ       → Error
(y - ŷ)²    → Squared error
MSE         → Average squared error

Regression  → Number
Classification → Category

Supervised  → X + y
Unsupervised → X only

Training    → Learn
Testing     → Evaluate unseen data

Overfitting → Too much fitting
Underfitting → Too little fitting

Goal → Generalize to unseen data


