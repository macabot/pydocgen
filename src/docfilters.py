"""
By Michael Cabot

Filters for removing unimportant parts from docstrings.
"""


from doctest import DocTestParser, Example
import logging
import re

logging.basicConfig(filename='./log.txt', level=logging.DEBUG, format='%(asctime)s %(message)s')

def remove_doctests(docstring):
    """Parse a docstring and remove doctest examples."""
    parser = DocTestParser()
    try:
        examples_and_strings = parser.parse(docstring)
        return ''.join(s for s in examples_and_strings if not isinstance(s, Example))
    except ValueError as e:
        logging.exception('\nValueError when removing doctest from:\n---\n%s\n---\n%s\n' % (docstring, e))
        return docstring

def test_remove_doctests():
    """test remove_doctests"""
    docstring = """
    belong a doctest
    >>> a = 3
    >>> a
    3

    end of doctest
    """
    print remove_doctests(docstring)

def remove_parameter_descriptions(docstring):
    """Remove parameter descriptions that start with a colon.
    E.g. ':type list sequence"""
    # TODO test
    lines = docstring.split('\n')
    good_lines = []
    for line in lines:
        stripped_line = line.strip()
        if not (len(stripped_line) > 0 and stripped_line[0] == ':'):
            good_lines.append(line)
    return '\n'.join(good_lines)

def remove_wx_wrappers(docstring):
    """Remove docstrings that are wrappers. Examples:
    IsOk(self) -> bool
    Focus(self, long index)
    """
    if re.search(r'\S+\(.*\)', docstring.strip()):
        return ''
    return docstring

def test_remove_wx_wrappers():
    """test remove_wx_wrappers"""
    assert remove_wx_wrappers("""IsOk(self) -> bool""")==''
    assert remove_wx_wrappers("""Focus(self, long index)""")==''
    assert remove_wx_wrappers("""Focus(self, long index""")!=''

def keep_first_description(docstring):
    """Remove everything after an empty line"""
    return re.split(r'\n\s*\n', docstring)[0]
    
def test_keep_first_description():
    """test keep_first_description"""
    docstring = """
    first description
    over multiple
    lines
    
    second
    
    third
    """
    print keep_first_description(docstring)
    print keep_first_description("""foo""")


if __name__ == '__main__':
    #test_remove_doctests()
    #test_remove_wx_wrappers()
    test_keep_first_description()
