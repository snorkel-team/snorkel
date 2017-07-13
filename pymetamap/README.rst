pymetamap
=========

Python wrapper around `MetaMap <http://metamap.nlm.nih.gov/>`_.
This will take a list of sentences and extract concepts using MetaMap
then return them in the form of a list of Concept objects.

**Note:** This code does **not** work with Windows because of my use of NamedTemporaryFile in SubprocessBackend.py.

How to Install
--------------

>>> python setup.py install

Example Usage
-------------

To start you must create a MetaMap instance from the pymetamap package.
::
    >>> from pymetamap import MetaMap
    >>> mm = MetaMap.get_instance('/opt/public_mm/bin/metamap12')

You must supply the metamap binary to ``get_instance()`` in order to
extract concepts.

To extract concepts from a sentence use the ``extract_concepts()``
method. This method taks a list of sentences as input and will return
a list of Concept objects.
::
    >>> sents = ['Heart Attack', 'John had a huge heart attack']
    >>> concepts,error = mm.extract_concepts(sents,[1,2])
    >>> for concept in concepts:
    ...     print concept
    Concept(index='1', mm='MM', score='14.64', preferred_name='Myocardial Infarction', cui='C0027051', semtypes='[dsyn]', trigger='["Heart attack"-tx-1-"Heart Attack"]', location='TX', pos_info='1:12', tree_codes='C14.280.647.500;C14.907.585.500')
    Concept(index='2', mm='MM', score='13.22', preferred_name='Myocardial Infarction', cui='C0027051', semtypes='[dsyn]', trigger='["Heart attack"-tx-1-"heart attack"]', location='TX', pos_info='17:12', tree_codes='C14.280.647.500;C14.907.585.500')

This example shows two seperate concepts extracted via MetaMap from two
different sentences (sentence 1 and sentence 2).

More Information
----------------

Licensed under `Apache 2.0 <http://www.apache.org/licenses/LICENSE-2.0>`_.

Written by Anthony Rios

Special thanks to `joaopalotti <https://github.com/joaopalotti>`_ for his contributions.
