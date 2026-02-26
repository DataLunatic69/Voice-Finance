#!/usr/bin/env bash

# Financial Analysis System - Setup & Debug Guide
# This script helps you set up and troubleshoot the system

echo ""
echo "=========================================="
echo "📊 FINANCIAL ANALYSIS SYSTEM - Setup Guide"
echo "=========================================="
echo ""

# Check Python
echo "✓ Checking Python..."
.venv/bin/python --version || echo "❌ Python not found in .venv"

# Check if .env exists
echo ""
echo "✓ Checking API Configuration..."
if [ -f ".env" ]; then
    if grep -q "GROQ_API_KEY=" .env; then
        GROQ_KEY=$(grep "GROQ_API_KEY=" .env | cut -d'=' -f2)
        if [ -z "$GROQ_KEY" ] || [ "$GROQ_KEY" = "your_groq_api_key_here" ]; then
            echo "❌ GROQ_API_KEY is not configured in .env"
            echo "   Please update it with your actual API key"
        else
            echo "✅ GROQ_API_KEY is configured"
        fi
    else
        echo "❌ GROQ_API_KEY not found in .env"
    fi
else
    echo "❌ .env file not found"
    echo "   Run: cp .env.example .env"
fi

# Check dependencies
echo ""
echo "✓ Checking dependencies..."
.venv/bin/python -c "import langchain; print('✅ langchain')" 2>/dev/null || echo "❌ langchain not installed"
.venv/bin/python -c "import langgraph; print('✅ langgraph')" 2>/dev/null || echo "❌ langgraph not installed"
.venv/bin/python -c "import streamlit; print('✅ streamlit')" 2>/dev/null || echo "❌ streamlit not installed"

# List available Groq models info
echo ""
echo "=========================================="
echo "🔑 API KEY SETUP INSTRUCTIONS"
echo "=========================================="
echo ""
echo "1. Get a FREE Groq API key:"
echo "   → Visit: https://console.groq.com/keys"
echo "   → No credit card required!"
echo "   → Copy your API key"
echo ""
echo "2. Add to .env file:"
echo "   → Edit the .env file in this directory"
echo "   → Find: GROQ_API_KEY=your_groq_api_key_here"
echo "   → Replace with your actual key"
echo ""
echo "3. (Optional) Get stock data API:"
echo "   → Visit: https://www.alphavantage.co/api/"
echo "   → Free tier available"
echo "   → Add to ALPHAVANTAGE_API_KEY in .env"
echo ""
echo "=========================================="
echo "🚀 USAGE EXAMPLES"
echo "=========================================="
echo ""
echo "Command line usage:"
echo "  .venv/bin/python -m src.main --query 'Analyze AAPL stock'"
echo ""
echo "Interactive usage:"
echo "  .venv/bin/python -m src.main"
echo ""
echo "Web UI (Streamlit):"
echo "  .venv/bin/streamlit run src/ui/app.py"
echo ""
echo "View logs:"
echo "  tail -f financial_analysis.log"
echo ""
echo "=========================================="
echo ""
