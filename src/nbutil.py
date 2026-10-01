"""Utilidades minimas para construir notebooks .ipynb programaticamente."""
import nbformat as nbf

def new_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3 (lending-club)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    nb.cells = []
    return nb

def add_md(nb, text):
    nb.cells.append(nbf.v4.new_markdown_cell(text))

def add_code(nb, code):
    nb.cells.append(nbf.v4.new_code_cell(code))

def save(nb, path):
    nbf.write(nb, path)
