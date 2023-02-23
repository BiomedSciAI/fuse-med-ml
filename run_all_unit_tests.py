"""
This script searches recursively for unit tests and generates the tests results in xml format in the way that Jenkins expects.
In the case that it's a Jenkins job, it should delete any created cache (not implemented yet)
"""

import logging
import sys
import unittest
import os
import termcolor
import xmlrunner

print(os.path.dirname(os.path.realpath(__file__)))


def mehikon(a, b):
    print(a)


termcolor.cprint = mehikon  # since junit/jenkins doesn't like text color ...

if __name__ == "__main__":
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[1]  # options "examples", "core" or None for both "core" and "examples"
    os.environ["DISPLAY"] = ""  # disable display in unit tests

    is_jenkins_job = "WORKSPACE" in os.environ and len(os.environ["WORKSPACE"]) > 2

    search_base = os.path.dirname(os.path.realpath(__file__))
    output = f"{search_base}/test-reports/"
    print("will generate unit tests output xml at :", output)

    sub_sections_core = [
        ("fuse/dl", search_base),
        ("fuse/eval", search_base),
        ("fuse/utils", search_base),
        ("fuse/data", search_base),
    ]
    sub_sections_fuseimg = [("fuseimg", search_base)]
    sub_sections_examples = [("fuse_examples/tests", search_base)]
    if mode is None:
        sub_sections = sub_sections_core + sub_sections_fuseimg + sub_sections_examples
    elif mode == "core":
        sub_sections = sub_sections_core
    elif mode == "fuseimg":
        sub_sections = sub_sections_fuseimg
    elif mode == "examples":
        sub_sections = sub_sections_examples
    else:
        raise Exception(f"Error: unexpected mode {mode}")

    suite = None
    for curr_subsection, top_dir in sub_sections:
        curr_subsuite = unittest.TestLoader().discover(
            f"{search_base}/{curr_subsection}", "test*.py", top_level_dir=top_dir
        )
        if suite is None:
            suite = curr_subsuite
        else:
            suite.addTest(curr_subsuite)

    # enable fuse logger and avoid colors format
    lgr = logging.getLogger("Fuse")
    lgr.setLevel(logging.INFO)

    test_results = xmlrunner.XMLTestRunner(output=output, verbosity=2, stream=sys.stdout).run(
        suite,
    )
