# SHL Assessment Recommender

## Overview

The SHL Assessment Recommender is a conversational AI system designed to recommend SHL assessments based on user queries. It leverages a hybrid retrieval system combining BM25 and FAISS for efficient and accurate recommendations. The system is built using FastAPI and integrates advanced NLP techniques for conversational flow management.

## Features

- **Hybrid Retrieval System**: Combines BM25 and FAISS for high-quality recommendations.
- **Conversational Flow**: Handles clarifications, recommendations, refinements, and comparisons.
- **Optimized Retrieval**: Achieves high Recall@10 for better user satisfaction.
- **Deployment Ready**: Designed for deployment on free-tier platforms like Render and Railway.

## Project Structure

```
shl-assessment-agent/
├── app/
│   ├── main.py          # FastAPI entry point
│   ├── agent.py         # Core conversational logic
│   ├── retrieval.py     # Hybrid retrieval logic
│   ├── config.py        # Environment variable configuration
├── data/
│   ├── GenAI_SampleConversations/  # Sample conversations
│   ├── raw/
│       ├── shl_product_catalog.json  # Product catalog
├── scripts/
│   ├── evaluate_samples.py  # Evaluation script for Recall@10
├── .env                # Environment variables
├── README.md           # Project documentation
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Arun-kumar2206/shl-assessment-agent.git
   cd shl-assessment-agent
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the required variables (e.g., `GROQ_API_KEY`, `HF_TOKEN`).

## Usage

1. Start the FastAPI server:

   ```bash
   uvicorn app.main:app --reload
   ```

2. Access the API documentation:
   - Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) in your browser.

3. Interact with the `/chat` endpoint to get recommendations.

## Evaluation

To evaluate the retrieval system:

```bash
python scripts/evaluate_samples.py
```

## Deployment

The application is deployed and accessible at:
[SHL Assessment Recommender](https://shl-assessment-agent-production-8b95.up.railway.app/)

The application can be deployed on platforms like Railway. Ensure the following:

- Environment variables are configured in the platform's settings.
- Memory usage is optimized (e.g., lazy-loading FAISS, smaller embedding models).

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

## Acknowledgments

- **Libraries Used**: FastAPI, Sentence-Transformers, FAISS, BM25.
- **Deployment Platforms**: Railway.
- **Inspiration**: SHL assessments and conversational AI systems.

---
