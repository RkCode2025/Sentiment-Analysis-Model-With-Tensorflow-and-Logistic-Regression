# Sentiment Analysis Program

## Overview
This project is a sentiment analysis tool designed to predict the sentiment (positive or negative) of movie reviews. It leverages the IMDB dataset for training and testing, employs logistic regression as the classification model, and utilizes TensorFlow for efficient computation and model deployment.

## Features
- Processes and analyzes text data from the IMDB dataset.
- Predicts sentiments of movie reviews with high accuracy.
- Utilizes machine learning techniques, including logistic regression and TensorFlow frameworks.
- Preprocessing steps include tokenization, normalization, and vectorization of text data.

## Installation
To run this program locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/RkCode2025/Sentiment-Analysis-Model-With-Tensorflow-and-Logistic-Regression
   cd sentiment-analysis
   ```

2. **Set Up the Environment**:
   Create a Python virtual environment and activate it.
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Download the IMDB Dataset**:
   - The dataset is downloaded form (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Usage
1. Run the program to train the model and evaluate its performance:
   ```bash
   python sentiment_analysis.py
   ```
2. Modify the script to input your own movie reviews and predict sentiments.

## Project Structure
```
.
├── sentiment_analysis.py   # Main script for training and evaluation
├── README.md               # Project documentation
└── LICENSE                 # License information
```

## Examples
- **Training Output**:
  The program outputs the model's training and validation accuracy, allowing you to assess performance.
- **Prediction**:
  Input: *"This movie was absolutely fantastic!"
  Output: Positive sentiment

## Contributing
Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- [IMDB Dataset](https://www.kaggle.com/datasets)
- TensorFlow and Scikit-learn documentation

---
**Happy coding!**
