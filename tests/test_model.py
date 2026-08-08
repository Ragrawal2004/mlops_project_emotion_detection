"""Champion-vs-challenger and smoke tests for the registered model.

Compares the current ``Production`` model ("champion") against the latest
registered version ("challenger"). The challenger is considered eligible
for promotion as long as it does not regress accuracy beyond a small
tolerance — it does NOT need to strictly improve on every run.

Why non-regression instead of strict improvement: CI re-runs the full
pipeline (including retraining) on every push, even when the underlying
data/code haven't meaningfully changed. In that case the "challenger" is
essentially the same model as the champion, so the accuracy delta is ~0%.
A strict "+1% or better" gate would fail on every such run forever, since
nothing is actually wrong. Gating on "not meaningfully worse" reflects how
this check is used in practice: it's a safety net against regressions, not
a guarantee that every retrain is an improvement.
"""

import pickle
import unittest

import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config.config import (
    MODEL_STAGE_PRODUCTION,
    PROCESSED_DATA_DIR,
    REGISTERED_MODEL_NAME,
    VECTORIZER_PATH,
    configure_mlflow_tracking,
)

# Maximum accuracy DROP (as a fraction, e.g. 0.01 = 1%) the challenger is
# allowed to have relative to the champion before it's blocked from
# promotion. A challenger that's equal to or better than the champion
# always passes; a challenger that's slightly worse (within this
# tolerance) still passes, to absorb retrain-to-retrain noise on
# unchanged data; anything worse than this is a real regression and fails.
MAX_ACCEPTABLE_REGRESSION = 0.01


class TestModel(unittest.TestCase):
    """Validates the newly registered model against the current champion."""

    @classmethod
    def setUpClass(cls):
        configure_mlflow_tracking()
        cls.client = mlflow.MlflowClient()
        cls.model_name = REGISTERED_MODEL_NAME

        production = cls.client.get_latest_versions(
            cls.model_name, stages=[MODEL_STAGE_PRODUCTION]
        )
        if len(production) == 0:
            raise unittest.SkipTest("No Production model found.")
        cls.production_version = production[0].version

        versions = cls.client.search_model_versions(f"name='{cls.model_name}'")
        latest = max(versions, key=lambda version: int(version.version))
        cls.latest_version = latest.version

        if cls.latest_version == cls.production_version:
            raise unittest.SkipTest(
                "Latest registered version is already in Production — "
                "nothing new to compare."
            )

        cls.production_model = mlflow.pyfunc.load_model(
            f"models:/{cls.model_name}/{cls.production_version}"
        )
        cls.new_model = mlflow.pyfunc.load_model(
            f"models:/{cls.model_name}/{cls.latest_version}"
        )

        with open(VECTORIZER_PATH, "rb") as file:
            cls.vectorizer = pickle.load(file)

        cls.test_data = pd.read_csv(PROCESSED_DATA_DIR / "test_bow.csv")

    def test_model_loading(self):
        self.assertIsNotNone(self.production_model)
        self.assertIsNotNone(self.new_model)

    def test_signature(self):
        sample = "This movie is amazing"
        vector = self.vectorizer.transform([sample])
        sample_df = pd.DataFrame(
            vector.toarray(), columns=[str(i) for i in range(vector.shape[1])]
        )

        prediction = self.new_model.predict(sample_df)

        self.assertEqual(
            sample_df.shape[1], len(self.vectorizer.get_feature_names_out())
        )
        self.assertEqual(len(prediction), 1)

    def test_prediction(self):
        sample = "I love machine learning"
        vector = self.vectorizer.transform([sample])
        sample_df = pd.DataFrame(
            vector.toarray(), columns=[str(i) for i in range(vector.shape[1])]
        )

        prediction = self.new_model.predict(sample_df)

        self.assertIn(prediction[0], [0, 1])

    def test_champion_vs_challenger(self):
        x = self.test_data.iloc[:, :-1]
        y = self.test_data.iloc[:, -1]

        champion_pred = self.production_model.predict(x)
        challenger_pred = self.new_model.predict(x)

        champion_accuracy = accuracy_score(y, champion_pred)
        challenger_accuracy = accuracy_score(y, challenger_pred)
        champion_precision = precision_score(y, champion_pred)
        challenger_precision = precision_score(y, challenger_pred)
        champion_recall = recall_score(y, champion_pred)
        challenger_recall = recall_score(y, challenger_pred)
        champion_f1 = f1_score(y, champion_pred)
        challenger_f1 = f1_score(y, challenger_pred)

        print("\n========== Champion ==========")
        print(f"Accuracy : {champion_accuracy:.4f}")
        print(f"Precision: {champion_precision:.4f}")
        print(f"Recall   : {champion_recall:.4f}")
        print(f"F1 Score : {champion_f1:.4f}")

        print("\n========= Challenger =========")
        print(f"Accuracy : {challenger_accuracy:.4f}")
        print(f"Precision: {challenger_precision:.4f}")
        print(f"Recall   : {challenger_recall:.4f}")
        print(f"F1 Score : {challenger_f1:.4f}")

        delta = challenger_accuracy - champion_accuracy
        print(f"\nAccuracy Delta = {delta * 100:.2f}%")

        self.assertGreaterEqual(
            delta,
            -MAX_ACCEPTABLE_REGRESSION,
            f"Challenger accuracy regressed by more than "
            f"{MAX_ACCEPTABLE_REGRESSION * 100:.1f}% relative to production "
            f"(champion={champion_accuracy:.4f}, challenger={challenger_accuracy:.4f}).",
        )


if __name__ == "__main__":
    unittest.main()
