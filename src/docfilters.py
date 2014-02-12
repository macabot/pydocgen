from doctest import DocTestParser, Example
import logging

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