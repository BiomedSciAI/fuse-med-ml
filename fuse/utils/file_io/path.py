import os


def change_extension(filepath: str, new_extension: str) -> str:
    """
    modifies [filepath] extension to be [new_extension]

    for example _change_extension('/a/b/c.sdf/ew/some_file.zip', '7zip')
    will return '/a/b/c.sdf/ew/some_file.7zip'
    """

    _basename = os.path.basename(filepath)
    last_dot = _basename.rfind(".")
    if last_dot < 0:
        return filepath + "." + new_extension
    ans = os.path.join(os.path.dirname(filepath), _basename[:last_dot] + "." + new_extension)
    return ans


def get_extension(filepath: str) -> str:
    """
    Returns the extension.
    For example - get_extension('/a/b/c/d/banana.txt') will return '.txt'
    """
    _basename = os.path.basename(filepath)
    last_dot = _basename.rfind(".")
    if last_dot < 0:
        return ""
    ans = _basename[last_dot:]
    return ans


def remove_extension(filepath: str) -> str:
    """
    Returns the filename without the extension
    For example: remove_extension('/a/b/c/d/asdf.txt') with return '/a/b/c/d/asdf'
    """

    _dirname = os.path.dirname(filepath)
    _basename = os.path.basename(filepath)
    if "." not in _basename:
        return filepath

    last_dot = _basename.rfind(".")
    ans = os.path.join(_dirname, _basename[:last_dot])
    return ans
