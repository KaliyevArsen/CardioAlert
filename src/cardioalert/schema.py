"""One input contract for model training, CLI examples and the Telegram UI.

Codes follow the original UCI processed Heart Disease files, not Kaggle recodings.
The numeric bounds below are broad software input checks, not clinical cutoffs.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Real

import numpy as np
import pandas as pd

SCHEMA_VERSION = "uci-heart-original-v1"
DISCLAIMER = (
    "Educational research prototype. This model score is not a diagnosis or a "
    "validated personal probability of future disease. Do not use it to make "
    "treatment decisions; consult a qualified clinician for health concerns."
)


@dataclass(frozen=True)
class Field:
    name: str
    prompt: str
    minimum: float = 0
    maximum: float = 1000
    codes: tuple[int, ...] = ()
    integer: bool = False


FIELDS = (
    Field("age", "Age in years (18-120)", 18, 120, integer=True),
    Field("sex", "UCI sex code: 0 = female, 1 = male", codes=(0, 1)),
    Field("cp", "Chest pain TYPE: 1 = typical angina; 2 = atypical angina; "
          "3 = non-anginal pain; 4 = asymptomatic. These are not pain severity levels.",
          codes=(1, 2, 3, 4)),
    Field("trestbps", "Resting systolic blood pressure (mm Hg)", 40, 300),
    Field("chol", "Serum cholesterol (mg/dL)", 20, 1000),
    Field("fbs", "Fasting blood sugar > 120 mg/dL: 0 = no, 1 = yes", codes=(0, 1)),
    Field("restecg", "Resting ECG: 0 = normal; 1 = ST-T abnormality; "
          "2 = probable/definite left ventricular hypertrophy", codes=(0, 1, 2)),
    Field("thalach", "Maximum achieved heart rate (beats/minute)", 20, 300),
    Field("exang", "Exercise-induced angina: 0 = no, 1 = yes", codes=(0, 1)),
    Field("oldpeak", "ST depression induced by exercise relative to rest", -10, 20),
    Field("slope", "Peak exercise ST slope: 1 = upsloping; 2 = flat; 3 = downsloping",
          codes=(1, 2, 3)),
    Field("ca", "Number of major vessels colored by fluoroscopy (0-3)", codes=(0, 1, 2, 3)),
    Field("thal", "Thallium stress-test result: 3 = normal; 6 = fixed defect; "
          "7 = reversible defect. This field is not thalassemia.", codes=(3, 6, 7)),
)
FEATURES = [field.name for field in FIELDS]
FIELD_BY_NAME = {field.name: field for field in FIELDS}


def validate_value(name: str, value: object) -> float:
    field = FIELD_BY_NAME[name]
    if value is None:
        return np.nan
    if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value):
        raise ValueError(f"{name}: expected a finite number or null")
    value = float(value)
    if field.codes and value not in field.codes:
        raise ValueError(f"{name}: allowed codes are {field.codes}")
    if not field.codes and not field.minimum <= value <= field.maximum:
        raise ValueError(f"{name}: expected {field.minimum} to {field.maximum}")
    if field.integer and not value.is_integer():
        raise ValueError(f"{name}: expected an integer")
    return value


def validate_record(record: dict) -> pd.DataFrame:
    if not isinstance(record, dict):
        raise ValueError("Input must be a JSON object")
    missing = sorted(set(FEATURES) - record.keys())
    extra = sorted(record.keys() - set(FEATURES))
    if missing or extra:
        raise ValueError(f"Input keys do not match schema: missing={missing}, extra={extra}")
    values = {name: validate_value(name, record[name]) for name in FEATURES}
    if sum(pd.isna(v) for v in values.values()) > 3:
        raise ValueError("At most three explicit null values are allowed in a demo input")
    return pd.DataFrame([values], columns=FEATURES)
