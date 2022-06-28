import getpass
def ask_user(yes_no_question):
    res = ''
    while res not in ['y', 'n']:
        res = input(f'{yes_no_question}? [y/n]')
    return res =='y'

def get_fuse_examples_user_dir():
    return f'/projects/msieve_dev3/usr/{getpass.getuser()}/fuse_examples'
