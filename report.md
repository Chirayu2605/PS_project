# Binary Sentiment Analysis Using Perceptron (From Scratch)

## Project Report

---

## 1. What is this project?

This project checks whether a product review is **Positive** or **Negative**.

For example:
- "This phone is amazing!" --> **Positive**
- "Worst product ever" --> **Negative**

We built a simple machine learning model called **Perceptron** from scratch
using Python and NumPy. We did NOT use any ready-made ML library for training.

---

## 2. What is a Perceptron?

A Perceptron is the **simplest type of machine learning model**. Think of it
like a single brain cell that makes a yes/no decision.

### How does it work?

1. **Take inputs** (numbers representing words in a review)
2. **Multiply each input by a weight** (importance of that word)
3. **Add them all up**: `z = w1*x1 + w2*x2 + ... + bias`
4. **Make a decision**:
   - If z >= 0, predict **1** (Positive)
   - If z < 0, predict **0** (Negative)

### How does it learn?

When the model makes a **wrong prediction**, we fix the weights:
```
error = actual_answer - predicted_answer
new_weight = old_weight + learning_rate * error * input
new_bias   = old_bias   + learning_rate * error
```

- `learning_rate` (0.01) controls how big the corrections are
- We repeat this for many **epochs** (full passes through the data)

---

## 3. What is Bag of Words (BoW)?

Computers don't understand text, they only understand numbers. So we need to
convert text into numbers. We use the **Bag of Words** method:

### Steps:
1. **Build a vocabulary** -- collect all unique words from training reviews
2. **Convert each review to a vector** -- count how many times each word appears

### Example:
```
Vocabulary: {"good": 0, "bad": 1, "movie": 2}

Review: "good good movie"
Vector: [2, 0, 1]   (good=2 times, bad=0 times, movie=1 time)

Review: "bad movie"
Vector: [0, 1, 1]   (good=0 times, bad=1 time, movie=1 time)
```

### Limitation:
BoW ignores word ORDER. It treats "not good" same as "good not".

---

## 4. Project Structure (What file does what?)

```
PS_project/
|-- preprocessing.py        --> Cleans text (lowercase, remove punctuation)
|-- feature_extraction.py   --> Builds vocabulary, converts text to numbers
|-- perceptron.py           --> The Perceptron model (our ML algorithm)
|-- train.py                --> Connects everything, runs the pipeline
|-- evaluation.py           --> Checks accuracy, makes graphs
|-- main.py                 --> Run this to start the project
|-- generate_dataset.py     --> Creates the dataset (2200 reviews)
|-- dataset.csv             --> The actual data file
|-- accuracy_plot.png       --> Graph showing accuracy improvement
|-- report.md               --> This report
```

---

## 5. How the project runs (step by step)

```
Step 1: Load dataset.csv (2200 reviews)
         |
Step 2: Clean text (lowercase, remove punctuation)
         |
Step 3: Build vocabulary (list of all unique words)
         |
Step 4: Convert each review to a number vector (Bag of Words)
         |
Step 5: Split data: 80% for training, 20% for testing
         |
Step 6: Train the Perceptron on training data
         |
Step 7: Test on unseen data and check accuracy
         |
Step 8: Show results (accuracy, confusion matrix, sample predictions)
```

---

## 6. Dataset Details

| Detail          | Value                         |
|-----------------|-------------------------------|
| Total reviews   | 2200                          |
| Positive (1)    | 1100                          |
| Negative (0)    | 1100                          |
| File format     | CSV                           |
| Columns         | review (text), sentiment (0/1)|
| Training set    | 1760 (80%)                    |
| Testing set     | 440 (20%)                     |

---

## 7. Results

### Training Progress:
| Epoch | Accuracy |
|-------|----------|
| 1     | 93.07%   |
| 2     | 99.43%   |
| 3     | 99.77%   |
| 4     | 99.94%   |
| 6-15  | 100.00%  |

### Test Results:
| Metric           | Value   |
|------------------|---------|
| **Test Accuracy** | 100.00% |
| True Positives   | 210     |
| True Negatives   | 230     |
| False Positives  | 0       |
| False Negatives  | 0       |

### Confusion Matrix:
```
                 Predicted
               Neg(0)  Pos(1)
Actual Neg      230       0
Actual Pos        0     210
```

**What does this mean?**
- TP (210): Model correctly said "Positive" for positive reviews
- TN (230): Model correctly said "Negative" for negative reviews
- FP (0): Model never wrongly said "Positive" for a negative review
- FN (0): Model never wrongly said "Negative" for a positive review

---

## 8. Why is accuracy 100%?

Our dataset uses **clearly different words** for positive and negative reviews:
- Positive words: amazing, fantastic, excellent, love, great...
- Negative words: terrible, awful, horrible, hate, worst...

Since positive and negative reviews use completely different words, the
Perceptron can easily draw a line between them. This is called
**linearly separable** data.

In **real-world data** (like actual IMDB reviews), accuracy would be lower
(70-85%) because people use sarcasm, mixed opinions, and ambiguous language.

---

## 9. Limitations of Our Model

1. **Only works for simple cases** -- can't handle sarcasm ("oh great, another broken phone")
2. **Ignores word order** -- treats "not good" as two separate words
3. **Large vocabulary = slow** -- real datasets have thousands of words
4. **No confidence score** -- only says 0 or 1, not "75% positive"
5. **Single layer** -- can't learn complex patterns (unlike deep learning)

---

## 10. How to Run

```
# First time only: generate the dataset
python generate_dataset.py

# Run the project
python main.py
```

---

## 11. Libraries Used

| Library    | What it does               | Why we used it         |
|------------|----------------------------|------------------------|
| Python 3   | Main programming language  | Required by project    |
| NumPy      | Math operations, arrays    | For weights, vectors   |
| Pandas     | Read CSV files, tables     | For loading dataset    |
| Matplotlib | Create graphs/charts       | For accuracy plot      |
| re         | Regular expressions        | For removing punctuation|

**Note:** We did NOT use sklearn or any ML library for training the model.

---

## 12. Conclusion

We successfully built a binary sentiment classifier using a Perceptron
implemented from scratch. The model achieved 100% accuracy on our dataset.
This project helped us understand:

- How machine learning models learn from data
- How text is converted to numbers using Bag of Words
- How weights are updated to improve predictions
- The importance of training/testing splits

---

*Report by: [Your Name]*
*Course: E&TC Engineering, Third Year*
*Date: April 2026*
