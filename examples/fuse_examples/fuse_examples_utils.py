import os
def ask_user(yes_no_question):
    res = ''
    while res not in ['y', 'n']:
        res = input(f'{yes_no_question}? [y/n]')
    return res =='y'

def get_fuse_examples_user_dir():
    return os.path.join(os.environ['USER_HOME_PATH'],'fuse_examples')
