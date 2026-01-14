# Personal Blog Search Engine

A sophisticated search engine specifically designed to discover and index personal blog content across the web. This project combines web crawling, machine learning classification, and advanced search algorithms to create a focused search experience for personal blog content.

## 🌟 Features

- **Smart Web Crawler**: Efficiently crawls web pages to discover and collect blog content
- **ML-Powered Blog Classification**: Uses machine learning to identify and filter personal blog content
- **Advanced Search Capabilities**:
  - TF-IDF based relevance scoring
  - PageRank algorithm for result ranking
  - Smart snippet generation
  - Intelligent text preprocessing
- **Modern Web Interface**: Clean and responsive search interface
- **RESTful API**: Easy integration with other applications

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/garvit5555/personal_blog_search_engine.git
cd personal_blog_search_engine
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Unix or MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

The application can be run in different modes depending on your needs:

1. **Full Pipeline** (Crawling, Training, and Search):
```bash
python run.py --crawl --train
```

2. **Just the Search Engine** (using existing data and model):
```bash
python run.py
```

3. **Custom Configuration**:
```bash
python run.py --crawl --train --max_pages 2000 --port 8000
```

Available command-line arguments:
- `--crawl`: Run the web crawler
- `--train`: Train the classifier
- `--max_pages`: Maximum number of pages to crawl (default: 1000)
- `--data_dir`: Directory to store crawled data (default: 'data')
- `--host`: Host to run the search engine on (default: '0.0.0.0')
- `--port`: Port to run the search engine on (default: 5000)

## 🏗️ Project Structure

```
personal_blog_search_engine/
├── data/                   # Crawled and processed data
├── index/                  # Search index files
├── models/                # Trained ML models
├── src/
│   ├── classifier/        # Blog classification module
│   ├── crawler/          # Web crawling module
│   ├── frontend/         # Web interface
│   └── search_engine/    # Core search functionality
├── requirements.txt       # Project dependencies
└── run.py                # Main entry point
```

## 🛠️ Technical Details

### Web Crawler
- Built with `requests` and `BeautifulSoup4`
- Respects robots.txt and implements polite crawling
- Extracts content, metadata, and link structure

### Blog Classifier
- Uses scikit-learn for machine learning
- Implements hierarchical classification
- Filters non-blog content and non-personal blogs

### Search Engine
- TF-IDF vectorization for content representation
- PageRank implementation for result ranking
- NLTK-based text preprocessing
- Efficient index storage and retrieval

## 📚 Dependencies

Key dependencies include:
- beautifulsoup4==4.12.2
- requests==2.31.0
- scikit-learn==1.3.2
- flask==2.3.3
- transformers==4.36.2
- nltk==3.8.1
- networkx==3.2.1

For a complete list, see `requirements.txt`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

- Garv Juneja
- GitHub: [@GarvJuneja977](https://github.com/GarvJuneja977)

## 🙏 Acknowledgments

- Thanks to all contributors who have helped shape this project.
- Special thanks to the open-source community for the amazing tools and libraries that made this possible.
