#!/usr/bin/env python
"""Simple launcher that sets DEBUG=true and runs main.py"""
import os
import sys

# Set DEBUG environment variable
os.environ['DEBUG'] = 'true'

# Change to project directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Import and run main
sys.path.insert(0, os.getcwd())
from src.main import main

if __name__ == '__main__':
    main()
