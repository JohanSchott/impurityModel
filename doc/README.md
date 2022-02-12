This directory holds the documentation for the impurityModel software.

## Generating HTML

    make -C sphinx html

The output will be in `sphinx/generated_doc/html`.

## How to Contribute

### File Formats

Add documentation text to existing .rst files in this folder.
Or create new .rst files and include them in the Sphinx documentation by including them
in the index.rst file.

### Rendering

Sphinx is used to render all documentation as one unit, including .py
docstrings.

For individual files, GitLab, Visual Studio Code, or some other text editor
can be used.

