"""
Anomalyze - Vercel Serverless Function Entry Point
"""
from __future__ import annotations
import sys
from pathlib import Path

current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from app_vercel import app