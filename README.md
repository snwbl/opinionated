# Advanced Aspect-Based Sentiment Analysis System

A comprehensive sentiment analysis system that performs advanced aspect-based sentiment analysis with multiple features including aspect extraction, contextual analysis, and visualization capabilities.

## Features

- Advanced aspect extraction using dependency parsing and BERT
- Contextual sentiment analysis with multiple methods (VADER and transformers)
- Aspect-opinion pair extraction
- Advanced analytical capabilities including clustering and comparison
- Comprehensive visualization tools
- Reporting and export functions

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
sentiment_analysis/
├── src/
│   ├── aspect_extraction/
│   ├── sentiment_analysis/
│   ├── visualization/
│   └── utils/
├── tests/
├── examples/
└── docs/
```

## Usage

```python
from sentiment_analysis import AspectSentimentAnalyzer

# Initialize the analyzer
analyzer = AspectSentimentAnalyzer()

# Analyze text
results = analyzer.analyze("The camera quality is excellent but the battery life is poor.")

# Get aspect-based sentiment
aspect_sentiments = analyzer.get_aspect_sentiments(results)
```

## Requirements

- Python 3.8+
- See requirements.txt for full list of dependencies

## License

MIT License
