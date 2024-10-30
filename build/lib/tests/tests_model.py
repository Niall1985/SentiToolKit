import unittest
from sentitoolkit.Main_tensor_model import SentiToolKit

class TestSentiToolKit(unittest.TestCase):

    def setUp(self):
      
        self.toolkit = SentiToolKit()

    def test_positive_sentiment(self):
        result = self.toolkit("This product is amazing!")
        self.assertEqual(result, 'Positive')

    def test_negative_sentiment(self):
        result = self.toolkit("This product is terrible!")
        self.assertEqual(result, 'Negative')

    def test_neutral_sentiment(self):
        result = self.toolkit("The product is okay.")
        self.assertEqual(result, 'Neutral')

    def test_empty_input(self):
        result = self.toolkit("")
        self.assertEqual(result, 'Neutral')  

if __name__ == '__main__':
    unittest.main()
