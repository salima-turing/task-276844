import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class TestEnsembleModelConsistency(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize the ensemble model
        self.num_runs = 5
        self.ensemble_models = [RandomForestClassifier(random_state=i) for i in range(self.num_runs)]

    def test_ensemble_model_consistency(self):
        """
        Verify the consistency of the ensemble model's output over multiple runs.
        """
        consistency_threshold = 0.95  # Set a desired consistency threshold

        for model in self.ensemble_models:
            model.fit(self.X_train, self.y_train)

        # Predict on the test set using each model
        predictions = [model.predict(self.X_test) for model in self.ensemble_models]

        # Calculate the average prediction
        average_prediction = np.mean(predictions, axis=0)

        # Calculate the consistency score
        consistency_score = np.mean(predictions == average_prediction)

        self.assertGreaterEqual(consistency_score, consistency_threshold,
                                msg=f"Ensemble model consistency score is below the threshold: {consistency_score:.2f} < {consistency_threshold:.2f}")


if __name__ == '__main__':
    unittest.main()
