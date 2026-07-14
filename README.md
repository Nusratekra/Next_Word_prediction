# Next Word Prediction

A comparative study of next-word prediction techniques, ranging from classical statistical language models to deep learning architectures, trained on the **BookCorpus** dataset (50k samples).

## Approaches Implemented

**Classical / Statistical**
- **Bigram model** — predicts the next word based on the previous single word
- **Trigram model** — predicts the next word based on the previous two words

**Machine Learning**
- **Naive Bayes (ComplementNB)** with TF-IDF features

**Deep Learning** (built with TensorFlow/Keras)
- **Simple RNN** — `Embedding → SimpleRNN → Dropout → Dense(softmax)`
- **GRU** — `Embedding → GRU → Dropout → Dense(softmax)`
- **LSTM** — `Embedding → LSTM → Dropout → Dense(softmax)`

All deep learning models are trained with the Adam optimizer and sparse categorical cross-entropy loss, and compared on:
- Test accuracy
- Training time
- Perplexity

## Dataset
- `bookcorpus_50k.csv` — a 50,000-sample subset of the BookCorpus dataset used for training and evaluating all models.

## Workflow
1. Text cleaning and preprocessing
2. Tokenization and padding (Keras `Tokenizer` + `pad_sequences`)
3. Train/test split
4. Train bigram, trigram, Naive Bayes, RNN, GRU, and LSTM models
5. Evaluate and compare accuracy, training time, and perplexity across all models with bar charts and accuracy/loss curves
6. Interactive prediction: enter a phrase and get next-word predictions from each model side by side
7. Save trained models (`.keras` for deep learning models, `.pkl` for classical/ML models) for later reuse

## Tech Stack
- **Language:** Python (Jupyter Notebook)
- **Deep Learning:** TensorFlow / Keras
- **Machine Learning:** scikit-learn (TF-IDF, ComplementNB)
- **Data Handling:** pandas, numpy
- **Visualization:** matplotlib

## How to Run
1. Open `next-word-prediction.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Install dependencies: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`.
3. Ensure `bookcorpus_50k.csv` is in the same directory (or update the file path).
4. Run all cells in order to train all six models and view the comparison charts.
5. Use the interactive input cell at the end to type a phrase and see next-word predictions from each model.

## Files
```
Next_Word_prediction/
├── bookcorpus_50k.csv
├── next-word-prediction.ipynb
