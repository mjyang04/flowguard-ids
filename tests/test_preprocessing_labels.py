"""Torch-independent tests for label encoding logic."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nids.data.preprocessing import (
    LABEL_MAPPING_MULTI,
    LABEL_MAPPING_UNSW_MULTI,
    prepare_labels,
)


def test_unsw_multiclass_has_ten_classes():
    assert len(set(LABEL_MAPPING_UNSW_MULTI.values())) == 10
    assert LABEL_MAPPING_UNSW_MULTI["normal"] == 0


def test_unsw_multiclass_mapping_is_stable_across_calls():
    df1 = pd.DataFrame(
        {"attack_cat": ["Normal", "Generic", "DoS", "Exploits", "Worms"]}
    )
    df2 = pd.DataFrame({"attack_cat": ["Worms", "Normal", "Generic"]})
    y1 = prepare_labels(df1, "unsw_nb15", label_mode="multiclass")
    y2 = prepare_labels(df2, "unsw_nb15", label_mode="multiclass")
    code = dict(zip(df1["attack_cat"].str.lower(), y1))
    assert code["worms"] == y2[0]
    assert code["normal"] == y2[1]
    assert code["generic"] == y2[2]


def test_unsw_multiclass_unknown_label_raises():
    df = pd.DataFrame({"attack_cat": ["Normal", "FuturisticMalware"]})
    with pytest.raises(ValueError, match="Unknown UNSW"):
        prepare_labels(df, "unsw_nb15", label_mode="multiclass")


def test_cicids_multiclass_mapping_covers_fifteen_classes():
    assert len(set(LABEL_MAPPING_MULTI.values())) == 15
    assert LABEL_MAPPING_MULTI["benign"] == 0
