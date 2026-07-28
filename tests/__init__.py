"""
Marks `tests` as a package.

Present so the shared doubles in `conftest` can be imported as
`tests.conftest` rather than relying on pytest's sys.path insertion, which makes
the import unambiguous when a test module and a source module share a name.
"""
