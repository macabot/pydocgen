"""
By Michael Cabot

Useful tools.
"""

import os

def uniform_string(string, max_chars = 60, prefix = '...'):
    """Make a string of the required length by adding white space or by
    slicing"""
    string_length = len(string)
    if string_length <= max_chars:
        return string + ' ' * (max_chars - string_length)
    return '...'  + string[len(prefix)+string_length - max_chars:]

def test_uniform_string():
    """Test uniform_string"""
    print uniform_string('a string', 10) + '|'
    print uniform_string('a long string', 10) + '|'

def iter_files_with_extension(path, extension):
    """Yield all files in a path with the given extension."""
    for root, _dirs, files in os.walk(path):
        for name in files:
            _name_root, ext = os.path.splitext(name)
            if ext == extension:
                yield os.path.join(root, name)

def iter_zipfiles_with_extension(zip_file, extension):
    """Find files in zip with the given extension."""
    for path in zip_file.namelist():
        _root, ext = os.path.splitext(path)
        if ext == extension:
            yield path


if __name__ == '__main__':
    test_uniform_string()