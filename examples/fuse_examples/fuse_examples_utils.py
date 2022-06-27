def ask_user(yes_no_question):
    res = ''
    while res not in ['y', 'n']:
        res = input(f'{yes_no_question}? [y/n]')
    return res =='y'
