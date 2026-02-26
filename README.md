# 📊 Financial Analysis System - Text Input Only

AI-powered financial market analysis system using **Groq's FREE LLM API** with text input.

## ✨ Features

✅ **Text Input Only** - No voice processing  
✅ **Groq LLM Integration** - 100% FREE API, no credit card needed  
✅ **Multi-Source Data** - Price and news data integration  
✅ **Vector Search** - Semantic document retrieval with embeddings  
✅ **Professional Reports** - Executive summaries with recommendations  
✅ **Detailed Logging** - Comprehensive execution logs for debugging  
✅ **Web UI** - Streamlit interface for easy interaction  
✅ **CLI Support** - Command-line interface for automation  

## 🚀 Quick Start

### 1. Get Free Groq API Key
Visit: https://console.groq.com/keys (no credit card needed!)

### 2. Configure
```bash
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

### 3. Run
```bash
# CLI
.venv/bin/python -m src.main --query "Analyze AAPL stock"

# Web UI
.venv/bin/streamlit run src/ui/app.py

# Interactive
.venv/bin/python -m src.main
```

## 📊 How It Works

1. **Extract Keywords** - Identifies companies, sectors, financial terms
2. **Fetch Data** - Retrieves price and news information  
3. **Vector Search** - Finds relevant documents using embeddings
4. **Analyze** - Uses Groq LLM to analyze trends and sentiment
5. **Report** - Generates executive summary with recommendations

## 🔑 API Configuration

**Groq API:**
- Website: https://console.groq.com/
- Cost: **100% FREE**
- Rate Limit: 30 requests/minute
- Model: `mixtral-8x7b-32768`

## 📁 Key Files

- `src/main.py` - CLI entry point
- `src/core/llm_config.py` - Groq LLM setup
- `src/ui/app.py` - Streamlit web interface
- `.env.example` - Configuration template
- `financial_analysis.log` - Execution logs

## 📚 Documentation

- **QUICKSTART.md** - Quick reference guide
- **SETUP_COMPLETE.md** - Full setup instructions
- **FINAL_SUMMARY.txt** - Project summary
- **CHANGES_MADE.md** - Modifications made

## 💡 Example Usage

```bash
# Simple query
.venv/bin/python -m src.main --query "Analyze Apple stock"

# Interactive mode
.venv/bin/python -m src.main

# Web UI
.venv/bin/streamlit run src/ui/app.py
```

## 🐛 Debugging

View logs in real-time:
```bash
tail -f financial_analysis.log
```

## ✨ What's Changed

✅ Removed OpenAI completely  
✅ Using Groq free API instead  
✅ Text-only input (no voice)  
✅ Added comprehensive logging  
✅ Fixed all workflow issues  
✅ Ready for production use  

## 📦 Requirements

- Python 3.10+
- Groq API key (FREE)
- 5 minutes to setup

## 🎯 Next Steps

1. Get Groq API key: https://console.groq.com/keys
2. Add to .env file
3. Run the app!

Status: ✅ **FULLY FUNCTIONAL**
- 🧠 **AI Analysis** - Powered by Groq/Llama 70B for intelligent insights
- 🔍 **Vector Search** - ChromaDB for semantic document retrieval
- 📊 **Interactive Dashboard** - Real-time charts and visualizations
- 🏗️ **Microservices** - Modular agent-based architecture
- 🐳 **Docker Ready** - Containerized deployment

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- API Keys: OpenAI, Groq, AlphaVantage

### Local Setup
```bash
# Clone repository
git clone https://github.com/yourusername/financial-analysis-system.git
cd financial-analysis-system

# Setup environment
cp .env.example .env
# Add your API keys to .env file

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run src/ui/app.py
```

### Using Microservices
```bash
# Start agents microservice
cd src/microservices
python main.py  # Runs on port 8001

# In another terminal, start main app
streamlit run src/ui/app.py  # Runs on port 8501
```

## 🌐 Deployment

### Streamlit Cloud : https://app-cloud-assistance-xibc2oaozz8ic6mbt8ryqk.streamlit.app/
    

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d --build

# Access services:
# Streamlit UI: http://localhost:8501
# Microservices API: http://localhost:8001
```

### Local Development
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📋 Configuration

| Environment Variable | Required | Description |
|---------------------|----------|-------------|
| `OPENAI_API_KEY` | ✅ | OpenAI API for Whisper speech processing |
| `GROQ_API_KEY` | ✅ | Groq Cloud API for LLM inference |
| `ALPHAVANTAGE_API_KEY` | ✅ | Stock market data from AlphaVantage |

## 🏗️ Architecture

```
src/
├── agents/              # Modular analysis agents
├── core/               # Shared models and configs  
├── microservices/      # API endpoints for agents
├── orchestration/      # LangGraph workflow
├── services/          # External API integrations
└── ui/                # Streamlit interface
```


**Key Demo Features:**
- Voice input recognition
- Real-time market data fetching
- AI-powered analysis generation
- Interactive dashboard visualization

## 🛠️ Usage Examples

### Voice Input
1. Click "🎤 Start Recording" in the UI
2. Speak your query: *"What's the latest on Apple stock?"*
3. Get comprehensive analysis with charts and insights

### Text Input
```python
# Example query
"Analyze Tesla stock performance and provide investment recommendations"
```


## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request
---
⭐ Star this repository if you find it helpful!
