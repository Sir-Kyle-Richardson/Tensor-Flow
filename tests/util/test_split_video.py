import unittest
from src.util.split_video import splitVideo
import os

currentPath = os.path.dirname(os.path.abspath(__file__))


class Test(unittest.TestCase):
    def testSplitVideoIntoFrames(self):
        videoPath = currentPath + "/../mocks/mock.mp4"
        result = splitVideo(videoPath)
        self.assertEqual(True, result)