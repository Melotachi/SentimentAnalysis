# Sentiment Analysis Project

A deep learning-based sentiment analysis system that analyzes customer reviews to determine sentiment (positive/negative) using state-of-the-art transformer models.

## Features

- Uses BERT-based transformer models for sentiment analysis
- Processes customer reviews in Turkish
- Provides sentiment scores and classifications
- Visualizes sentiment distribution
- Handles large datasets efficiently

## Technologies Used

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```
SentimentAnalysis/
├── sentimentanalysis.ipynb    # Main analysis notebook
├── reviews.csv                # Customer reviews dataset
└── README.md                  # Project documentation
```

## Setup

1. Install required packages:

```bash
pip install torch transformers pandas numpy matplotlib scikit-learn
```

2. Download the reviews dataset and place it in the project directory

## Usage

1. Open `sentimentanalysis.ipynb` in Jupyter Notebook
2. Run the cells in sequence to:
   - Load and preprocess the data
   - Initialize the sentiment analysis model
   - Analyze reviews and generate sentiment scores
   - Visualize results

## Model Details

- Uses a pre-trained BERT model fine-tuned for sentiment analysis
- Processes text in Turkish language
- Outputs sentiment scores and classifications

## Results

The analysis provides:

- Sentiment scores for each review
- Overall sentiment distribution
- Visual representations of the results

## License

This project is open source and available under the MIT License.
