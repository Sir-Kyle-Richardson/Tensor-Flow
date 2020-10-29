import unittest
from src.util.split_video import splitVideo
import os

class Test(unittest.TestCase):   
    def testSplitVideoIntoFrames(self):
        currentPath = os.path.realpath(__file__)
        videoPath = "../../src/"
        result = splitVideo("howdy")
        self.assertEqual(True, result)