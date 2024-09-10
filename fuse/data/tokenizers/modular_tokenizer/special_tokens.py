special_token_marker = [
    "<",
    ">",
]


def special_wrap_input(x: str) -> str:
    return special_token_marker[0] + x + special_token_marker[1]


def strip_special_wrap(x: str) -> str:
    for spec_wrap in special_token_marker:
        x = x.replace(spec_wrap, "")
    return x
