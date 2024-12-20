from unittest import TestCase
from autograd.tools.data import train_test_split
from autograd.tools.metrics import accuracy, precision
import numpy as np


class TestUtils(TestCase):
    def setUp(self) -> None:
        self.X = np.array(
            [
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
                [9, 10],
            ]
        )
        self.y = np.array([0, 1, 0, 1, 0])

        self.X_empty = np.array([])
        self.y_empty = np.array([])

    def test_train_test_split(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2
        )
        # We expect 80% of the data to be in the train set
        # We expect 20% of the data to be in the test set
        self.assertEqual(X_train.shape, (4, 2))
        self.assertEqual(X_test.shape, (1, 2))
        self.assertEqual(y_train.shape, (4,))
        self.assertEqual(y_test.shape, (1,))

        X_train, X_test, y_train, y_test = train_test_split(
            self.X_empty, self.y_empty, test_size=0.2
        )
        assert np.array_equal(X_train, self.X_empty)
        assert np.array_equal(X_test, self.X_empty)
        assert np.array_equal(y_train, self.y_empty)
        assert np.array_equal(y_test, self.y_empty)

    def test_accuracy(self):
        self.assertEqual(accuracy(self.y, self.y), 1.0)
        self.assertEqual(accuracy(self.y, 1 - self.y), 0.0)
        # The false (0) will be unchanged, so we will have 3/5 correct
        self.assertEqual(accuracy(self.y, self.y * 0.5), 0.6)

    def test_precision(self):
        self.assertEqual(precision(self.y, self.y), 1.0)

        # True positives = 0, False positives = 5, so precision = 0
        self.assertEqual(precision(self.y, 1 - self.y), 0.0)

        # True positive = 0, False positive = 0, so precision = 0, should handle division by zero
        self.assertEqual(precision(y_true=[1, 1, 1, 1, 1], y_pred=[0, 0, 0, 0, 0]), 0.0)

        # True positive = 2, False positive = 2, so precision = 0.5
        self.assertEqual(precision(y_true=[1, 1, 0, 0], y_pred=[1, 1, 1, 1]), 0.5)
