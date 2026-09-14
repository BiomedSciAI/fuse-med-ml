"""
This script searches recursively for unit tests and generates the tests results in xml format in the way that Jenkins expects.
In the case that it's a Jenkins job, it should delete any created cache (not implemented yet)
"""

import logging
import os
import sys

import pytest
import termcolor

print(os.path.dirname(os.path.realpath(__file__)))


def mehikon(a, b):  # type: ignore
    print(a)


termcolor.cprint = mehikon  # since junit/jenkins doesn't like text color ...

if __name__ == "__main__":
    mode = None
    if len(sys.argv) > 1:
        mode = sys.argv[
            1
        ]  # options "examples", "core" or None for both "core" and "examples"
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
        sub_sections = (
            sub_sections_core  # + sub_sections_fuseimg + sub_sections_examples
        )
    elif mode == "core":
        sub_sections = sub_sections_core
    elif mode == "fuseimg":
        sub_sections = sub_sections_fuseimg
    elif mode == "examples":
        sub_sections = sub_sections_examples
    else:
        raise Exception(f"Error: unexpected mode {mode}")

    # enable fuse logger and avoid colors format
    lgr = logging.getLogger("Fuse")
    lgr.setLevel(logging.INFO)

    # let pytest discover and run the tests - it natively collects
    # unittest.TestCase-based tests, so the existing tests are unaffected -
    # and produce the same kind of JUnit XML report that xmlrunner used to.
    os.makedirs(output, exist_ok=True)
    junit_report = os.path.join(output, "junit.xml")

    search_paths = [f"{search_base}/{curr_subsection}" for curr_subsection, _ in sub_sections]

    exit_code = pytest.main(
        [*search_paths, "-v", f"--junitxml={junit_report}"],
    )
    sys.exit(exit_code)
