from __future__ import absolute_import, division, print_function

import os
import shutil
import subprocess
import unittest
import sys


class TestCommonInstall(unittest.TestCase):
    def testDrakeFindResourceOrThrowInInstall(self):
        # Install into a temporary directory. The temporary directory does not
        # need to be removed as bazel tests are run in a scratch space.
        tmp_folder = "tmp"
        os.mkdir(tmp_folder)
        # Install the bindings and its dependencies in scratch space.
        subprocess.check_call(
            ["install",
             os.path.abspath(tmp_folder),
             ])
        content_install_folder = os.listdir(tmp_folder)
        # Check that `lib` and `share` folders are installed since they are
        # used in this test.
        self.assertIn('lib', content_install_folder)
        self.assertIn('share', content_install_folder)
        # Remove Bazel build artifacts, and ensure that we only have install
        # artifacts.
        content_test_folder = os.listdir(os.getcwd())
        content_test_folder.remove('tmp')
        for element in content_test_folder:
            if os.path.isdir(element):
                shutil.rmtree(element)
            else:
                os.remove(element)
        self.assertEqual(os.listdir("."), [tmp_folder])

        # Set the correct PYTHONPATH and (DY)LD_LIBRARY_PATH to use the
        # correct pydrake module.
        env_python_path = "PYTHONPATH"
        if sys.platform == 'darwin':
            env_ld_library = "DYLD_LIBRARY_PATH"
        else:
            env_ld_library = "LD_LIBRARY_PATH"
        tool_env = dict(os.environ)
        tool_env[env_python_path] = os.path.abspath(
            os.path.join(tmp_folder, "lib", "python2.7", "site-packages")
        )
        tool_env[env_ld_library] = os.path.abspath(
            os.path.join(tmp_folder, "lib")
        )
        data_folder = os.path.join(tmp_folder, "share", "drake", "drake")
        output_path = subprocess.check_output(
            ["python",
             "-c", "import pydrake; print(pydrake.getDrakePath())"
             ],
            env=tool_env,
            ).strip()
        found_install_path = (data_folder in output_path)
        self.assertEqual(found_install_path, True)


if __name__ == '__main__':
    unittest.main()