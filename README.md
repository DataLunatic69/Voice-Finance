# Financial Analysis System

![System Architecture](workflow_mermaid_diagram.png)

AI-powered system for real-time financial market analysis with voice and text input.

## Features

- 🎤 Voice input processing via OpenAI Whisper API
- 📈 Multi-source data aggregation (AlphaVantage, Yahoo Finance)
- 🧠 AI-powered analysis using Groq/Llama 70B
- 🔍 Vector search with ChromaDB
- 📊 Interactive Streamlit dashboard
- 🐳 Docker-ready deployment

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/financial-analysis-system.git
cd financial-analysis-system

# Setup environment
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -e .[dev]

# Run Streamlit UI
streamlit run src/ui/app.py

# Or run CLI version
python -m financial-analysis --query "Analyze AAPL stock"
```

## Deployment

### Docker
```bash
docker-compose up -d --build
```

### Cloud Deployment
```bash
# Example for AWS ECS
aws ecr create-repository --repository-name financial-analysis
docker build -t financial-analysis .
docker tag financial-analysis:latest your-account-id.dkr.ecr.region.amazonaws.com/financial-analysis:latest
docker push your-account-id.dkr.ecr.region.amazonaws.com/financial-analysis:latest
```

## Configuration

| Environment Variable      | Required | Description                          |
|---------------------------|----------|--------------------------------------|
| `OPENAI_API_KEY`          | Yes      | OpenAI API key for speech processing |
| `GROQ_API_KEY`            | Yes      | Groq Cloud API key                   |
| `ALPHAVANTAGE_API_KEY`    | Yes      | AlphaVantage stock API key           |

