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

def remove_wx_wrappers(docstring):
    """Remove docstrings that are wrappers. Examples:
    IsOk(self) -> bool
    Focus(self, long index)
    """
    if re.search(r'\S+\(.*\)', docstring.strip(), flags=re.DOTALL):
        return ''
    return docstring

def test_remove_wx_wrappers():
    """test remove_wx_wrappers"""
    assert remove_wx_wrappers("""IsOk(self) -> bool""")==''
    assert remove_wx_wrappers("""Focus(self, long index)""")==''
    assert remove_wx_wrappers("""Focus(self, long index""")!=''
    text = """__init__(self, window parent, auimanager ownermgr, auipaneinfo pane, int id=id_any, 
    long style=wxresize_border|wxsystem_menu|wxcaption|wxframe_no_taskbar|wxframe_float_on_parent|wxclip_children) -> auifloatingframe"""
    assert remove_wx_wrappers(text)==''

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
    
def replace_vertical_bars(docstring):
    return re.sub(r'\|', 'BAR', docstring)
    
def test_replace_vertical_bars():
    assert replace_vertical_bars('implementation of | operator - returns matchfirst') == 'implementation of BAR operator - returns matchfirst'
    
def remove_parameter_descriptions(docstring):
    return re.sub(r':param \w+:.*', '', docstring)
    
def test_remove_parameter_descriptions():
    docstring = """specify the message data for topic messages. 
    :param argsdocs: a dictionary of keyword names (message data name) and data 'docstring'; cannot be none 
    :param required: a list of those keyword names, appearing in argsdocs, which are required (all others are assumed optional)"""
    assert remove_parameter_descriptions(docstring).strip() == 'specify the message data for topic messages.'


if __name__ == '__main__':
    #test_remove_doctests()
    #test_remove_wx_wrappers()
    #test_keep_first_description()
    #test_replace_vertical_bars()
    test_remove_parameter_descriptions()
