import ast
from pathlib import Path

def test_training_script_parses():
    ast.parse((Path(__file__).resolve().parents[1]/'train_custom_mlp.py').read_text())
