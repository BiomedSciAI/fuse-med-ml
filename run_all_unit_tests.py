'''
This script searches recursively for unit tests and generates the tests results in xml format in the way that Jenkins expects.
In the case that it's a Jenkins job, it should delete any created cache (not implemented yet)
'''

import logging
from sys import stdout
import sys
import unittest
import os
print(os.path.dirname(os.path.realpath(__file__)))
import termcolor
def mehikon(a,b):
    print(a)
termcolor.cprint = mehikon  #since junit/jenkins doesn't like text color ...
import xmlrunner

if __name__ == '__main__':
    os.environ['DISPLAY'] = '' #disable display in unit tests

    is_jenkins_job = 'WORKSPACE' in os.environ and len(os.environ['WORKSPACE'])>2

    search_base = os.path.dirname(os.path.realpath(__file__))
    output = f"{search_base}/test-reports/"
    print('will generate unit tests output xml at :',output)

    # with open(f'{search_base}/packages.txt','r') as f:
    #     sub_sections = [x.split('#')[-1].strip()+'/fuse/' for x in f.readlines() if len(x)>4]
    # print('found sub_sections = ', sub_sections)
    sub_sections = ["fuse/test", "fuse_examples/test"] 


    suite = None
    for curr_subsection in sub_sections:
        curr_subsuite = unittest.TestLoader().discover(
            f'{search_base}/{curr_subsection}', 'test*.py', top_level_dir=search_base
        )
        if suite is None:
            suite = curr_subsuite
        else:
            suite.addTest(curr_subsuite)
            
    # enable fuse logger and avoid colors format
    lgr = logging.getLogger('Fuse')
    lgr.setLevel(logging.INFO)

    test_results = xmlrunner.XMLTestRunner(output=output, verbosity=2, stream=sys.stdout).run(
        suite, 
    )
