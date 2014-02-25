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

def count_lines_with_extension(path, extension):
    """Count the number of lines of all files in the given path with the given
    extension"""
    return sum(number_of_lines(p) for p in utils.iter_files_with_extension(path, extension))

def num_lines_with_extension(path, extension):
    """Return a generator of tuples of the number of lines in a file and that
    file, given that the file has a certain extension"""
    return ((number_of_lines(p), p) for p in utils.iter_files_with_extension(path, extension))

def number_of_lines(path):
    """Count the number of lines in a file"""
    with open(path, 'r') as in_file:
        return sum(1 for _line in in_file)


if __name__ == '__main__':
    test_uniform_string()