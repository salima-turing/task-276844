import unittest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression


class TestEnsembleModelConsistency(unittest.TestCase):
    def setUp(self):
        # Create a simple regression dataset
        X, self.y = make_regression(n_samples=100, n_features=4, random_state=0)

        # Initialize the ensemble model
        self.ensemble_model = RandomForestRegressor(n_estimators=10, random_state=0)
        self.ensemble_model.fit(X, self.y)

    def test_output_consistency(self):
        """
        Verify that the ensemble model produces consistent output over multiple runs.
        """
        num_runs = 5
        tolerance = 0.01  # Adjust the tolerance as needed

        first_run_predictions = self.ensemble_model.predict(self.X)

        for _ in range(num_runs - 1):
            predictions = self.ensemble_model.predict(self.X)
            max_abs_diff = np.max(np.abs(first_run_predictions - predictions))
            self.assertLessEqual(max_abs_diff, tolerance, msg="Prediction output is inconsistent.")


if __name__ == '__main__':
    unittest.main()
