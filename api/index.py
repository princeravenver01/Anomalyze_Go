"""
Anomalyze - Vercel Serverless Function Entry Point
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add the current directory to Python path to ensure imports work
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app_vercel import app

# This is what Vercel's Python runtime looks for
# The app is imported directly from app_vercel in the same directory
