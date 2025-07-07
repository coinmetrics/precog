import unittest

from precog import __version__


class TestPackage(unittest.TestCase):

    # runs once prior to all tests in this file
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # runs once prior to every single test
    def setUp(self):
        pass

    def test_package_version(self):
        # Check that version is as expected
        # Must update to increment package version successfully
        self.assertEqual(__version__, "2.5.2")
