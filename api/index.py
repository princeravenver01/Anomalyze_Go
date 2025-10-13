"""
Anomalyze - Vercel Serverless Function Entry Point
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)

# Change working directory to parent so Flask can find templates
os.chdir(parent_dir)

from app_vercel import app

# This is what Vercel's Python runtime looks for
# Don't reassign, just use the imported app directly
