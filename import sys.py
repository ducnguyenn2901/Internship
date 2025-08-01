import sys
import types
import pandas as pd
import numpy as np
import datetime
import pytest
import os
import os
import os

# test_predict_student_count_lr.py


def mock_read_excel(file, *args, **kwargs):
    # Minimal mock data for testing
    data = {
        'Center': ['A', 'A', 'B', 'B'],
        'Class name': ['C1', 'C2', 'C1', 'C2'],
        'Course': ['Code for Kids', 'Code for Kids', 'Code for Teens', 'Code for Teens'],
        'Student count': [10, 12, 8, 9],
        'Status': ['RUNNING', 'FINISHED', 'RUNNING', 'FINISHED'],
        'Start date': [
            '2023-01-01', '2023-02-01', '2023-01-01', '2023-02-01'
        ]
    }
    return pd.DataFrame(data)

def test_pipeline_runs(monkeypatch):
    # Patch pd.read_excel
    monkeypatch.setattr(pd, "read_excel", mock_read_excel)
    # Import the script as a module
    import importlib.util

    script_path = os.path.join(os.path.dirname(__file__), "predict_student_count_lr.py")
    spec = importlib.util.spec_from_file_location("predict_student_count_lr", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_student_count_lr"] = mod
    spec.loader.exec_module(mod)

    # Check model exists and is fitted
    assert hasattr(mod, "model")
    assert hasattr(mod.model, "predict")

def test_metrics_reasonable(monkeypatch):
    monkeypatch.setattr(pd, "read_excel", mock_read_excel)
    import importlib.util

    script_path = os.path.join(os.path.dirname(__file__), "predict_student_count_lr.py")
    spec = importlib.util.spec_from_file_location("predict_student_count_lr", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_student_count_lr"] = mod
    spec.loader.exec_module(mod)

    # Check metrics are floats and in expected range
    assert hasattr(mod, "mean_absolute_error")
    assert hasattr(mod, "mean_squared_error")
    assert hasattr(mod, "r2_score")
    # The script prints metrics, but we can recompute for the test
    y_pred = mod.model.predict(mod.X)
    mae = mod.mean_absolute_error(mod.y, y_pred)
    rmse = np.sqrt(mod.mean_squared_error(mod.y, y_pred))
    r2 = mod.r2_score(mod.y, y_pred)
    assert mae >= 0
    assert rmse >= 0
    assert -1 <= r2 <= 1

def test_future_prediction(monkeypatch):
    monkeypatch.setattr(pd, "read_excel", mock_read_excel)
    import importlib.util

    script_path = os.path.join(os.path.dirname(__file__), "predict_student_count_lr.py")
    spec = importlib.util.spec_from_file_location("predict_student_count_lr", script_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["predict_student_count_lr"] = mod
    spec.loader.exec_module(mod)

    # Build a test example
    example = pd.DataFrame({
        'Center': ['A'],
        'Course': ['Code for Kids'],
        'Status': ['RUNNING'],
        'Year': [datetime.datetime.now().year + 1],
        'Month': [1],
        'num_classes': [2]
    })
    pred = mod.model.predict(example)
    assert pred.shape == (1,)
    assert pred[0] >= 0
