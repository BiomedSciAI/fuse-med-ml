'''
This script searches recursively for unit tests and generates the tests results in xml format in the way that Jenkins expects.
In the case that it's a Jenkins job, it should delete any created cache (not implemented yet)
'''

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
            

    test_results = xmlrunner.XMLTestRunner(output=output).run(
        suite, 
    )

    
#TODO: think about caches and how we deal with them in unit tests (where they are found and who is responsible to delete them)
#    if is_jenkins_job:
#        print('jenkins job is not supported yet, it requires a caching location etc.')
#        raise NotImplementedError

#        #TODO: delete the created cache
#        cache_path = caching.get_current_read_locations(caching_kind='user_local')
#        assert 1 == len(cache_path)
#        cache_path = cache_path[0]
#        assert 'ms_img_analytics' in cache_path
#        #delete the cache that we've created for this job
#        print(f'deleting eir cache {cache_path} ...')
#        os.system(f'rm -rf {cache_path}')
#        print('done deleting eir cache.')
