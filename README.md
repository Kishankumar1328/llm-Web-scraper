# Streamlit Web App for High-Quality Responses

## Overview

This project is a **Streamlit-based web application** that:

- Takes user input in the form of **URLs, links, etc.**
- Provides **high-quality answers** based on the given input
- Ensures **efficient responses** using a **powerful model (Ollama LLaMA 3 8B)**
- Uses **web scraping techniques** to extract relevant information
- Implements **vector search with ChromaDB** for contextual retrieval

## Features

- **User Input Handling:** Accepts URLs, links, or text as input.
- **Web Scraping:** Utilizes **Selenium** and **BeautifulSoup** to extract webpage content.
- **AI-Powered Responses:** Uses a **large-scale language model** (Ollama LLaMA 3 8B) for accurate and insightful answers.
- **Efficient Processing:** Optimized for **fast and reliable** response generation.
- **Vector Search:** Implements **ChromaDB** to store and retrieve text efficiently.
- **Interactive UI:** Built using **Streamlit** for a clean and user-friendly interface.

## Installation

Ensure you have Python installed, then set up your environment:

```sh
# Clone the repository
git clone https://github.com/your-repo/your-project.git
cd your-project

# Create and activate virtual environment
python -m venv venv  # Windows: Use `python -m venv venv`
source venv/bin/activate  # Windows: Use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Usage

Run the Streamlit app:

```sh
streamlit run app.py
```

## Dependencies

- **Streamlit** (For UI)
- **Selenium & BeautifulSoup** (For web scraping)
- **LangChain** (For text processing and retrieval)
- **ChromaDB** (For vector storage)
- **Ollama** (For running LLaMA 3 8B locally)

## Troubleshooting

### 1. `ModuleNotFoundError: No module named 'chromadb'`

Run:

```sh
pip install chromadb
```

### 2. `Chroma.__init__() got an unexpected keyword argument 'embeddings'`

Use `embedding` instead of `embeddings` in your code:

```python
vectorstore = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)
```

### 3. Running Ollama LLaMA 3 8B

Ensure you have **Ollama installed** and run:

```sh
ollama pull llama3:8b
ollama run llama3:8b
```

### 4. Selenium WebDriver Issues

If ChromeDriver fails, update it using:

```sh
pip install --upgrade webdriver-manager
```

Or specify the driver manually in `Options()`.

## Contributing

Feel free to fork this repository and submit **pull requests**. Suggestions and improvements are always welcome!

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for more details.

