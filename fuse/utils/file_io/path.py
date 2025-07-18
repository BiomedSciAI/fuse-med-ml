import re
from os.path import basename, dirname, join


def add_base_prefix(filepath: str, prefix: str) -> str:
    """
    Useful simple helper function for adding a prefix to a file. The prefix is added to the file "basename",
        and the full path is returned.
    Example usage:

        ans = add_base_prefix('/a/b/c/wow.tsv', 'extra_info@')
        print(ans)
        >'/a/b/c/extra_info@wow.tsv'


    """
    return join(dirname(filepath), prefix + basename(filepath))


def change_extension(filepath: str, new_extension: str) -> str:
    """
    Modifies [filepath] extension to be [new_extension]

    for example _change_extension('/a/b/c.sdf/ew/some_file.zip', '7zip')
    will return '/a/b/c.sdf/ew/some_file.7zip'
    """
    _basename = basename(filepath)
    last_dot = _basename.rfind(".")
    if last_dot < 0:
        return filepath + "." + new_extension
    ans = join(dirname(filepath), _basename[:last_dot] + "." + new_extension)
    return ans


def get_extension(filepath: str) -> str:
    """
    Returns the extension.
    For example - get_extension('/a/b/c/d/banana.txt') will return '.txt'
    """
    _basename = basename(filepath)
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
    _dirname = dirname(filepath)
    _basename = basename(filepath)
    if "." not in _basename:
        return filepath

    last_dot = _basename.rfind(".")
    ans = join(_dirname, _basename[:last_dot])
    return ans


def get_valid_filename(s: str) -> str:
    """
    Modifies an input string into a string that is valid as a filename in linux
    """
    s = str(s).strip().replace(" ", "_")
    return re.sub(r"[\\:\"*?<>|]", "@", s)
