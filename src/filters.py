from doctest import DocTestParser, Example

def remove_doctests(docstring):
    """Parse a docstring and remove doctest examples."""
    parser = DocTestParser()
    examples_and_strings = parser.parse(docstring)
    return ''.join(s for s in examples_and_strings if not isinstance(s, Example))