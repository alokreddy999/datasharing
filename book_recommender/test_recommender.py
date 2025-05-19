import os
import unittest
from recommender import BookRecommender


class TestBookRecommender(unittest.TestCase):
    def test_recommend_returns_results(self):
        data_path = os.path.join(os.path.dirname(__file__), 'books.csv')
        rec = BookRecommender(data_path)
        results = rec.recommend('dragon adventure')
        self.assertEqual(len(results), 5)


if __name__ == '__main__':
    unittest.main()
