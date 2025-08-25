import pandas as pd


def test_weight_class_change_updates_stats_features():
    weight_class_mapping = {"Featherweight": 0, "Lightweight": 1}
    stats_features_1 = pd.DataFrame({"Weight Class": [0]})
    stats_features_2 = pd.DataFrame({"Weight Class": [0]})

    label = "Lightweight"
    original_code = stats_features_1["Weight Class"].iloc[0]
    new_code = weight_class_mapping[label]

    stats_features_1["Weight Class"] = new_code
    stats_features_2["Weight Class"] = new_code

    assert new_code != original_code
    assert stats_features_1["Weight Class"].iloc[0] == new_code
    assert stats_features_2["Weight Class"].iloc[0] == new_code

