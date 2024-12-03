import unittest

from precog import __version__


class TestPackage(unittest.TestCase):

    def setUp(self):
        pass

    def test_package_version(self):
        # Check that version is as expected
        # Must update to increment package version successfully
        self.assertEqual(__version__, "0.1.0")
