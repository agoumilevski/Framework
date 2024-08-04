import os
from distutils.core import setup

path = os.path.dirname(os.path.abspath(__file__))
fpath = os.path.abspath(os.path.join(path,"../docs/README.md"))

NAME = 'Python Platform'
DESCRIPTION = 'Python Platform for DSGE Modeling'

with open(fpath) as f:
    long_description = f.read()

METADATA = dict(
    name=NAME, 
    version='1.0', 
    url='http://github.com/IMF/Framework/',
    license='GNU Lesser General Public License (LGPL)', 
    author='Alexei Goumilevski',
    author_email='Agoumilevski@imf.org',
    install_requires = [
      "pandas","scipy","numpy","datetime","tkinter", 
	  "matplotlib","dateutil","math","argparse",
	  "reportlab","sqlite3","sympy","symengine", 
      "emcee","statsmodels","ruamel.yaml","pydot",
      "networkx","particles","pymcmcstat","PyPDF2", 
      "corner","pylatex","textwrap","glob","seaborn",
      "ast","re","xlswriter","viztracer","numba",
      "sklearn","lark","graphviz","pydot","filterpy"
    ],
    description=DESCRIPTION, 
    long_description = long_description,
    long_description_content_type="text/markdown",
    packages=["src"], 
    include_package_data=True,
    data_files=[('api_docs', ['../api_docs']),
                ('bin', ['../bin']),
                ('data', ['../data']),  
                ('docs', ['../docs']),
                ('models', ['../models']),
                ('graphs', ['../graphs']),
                ('results', ['../results']),
                  ],
    platforms='any',
    classifiers=[
        'copyright :: Copyright 2024 International Monetary Fund. All rights reserved.',
        'company_name :: International Monetary Fund',
        'Programming Language :: Python',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Topic :: Macroeconomic Modeling :: DSGE',
        'Topic :: Software Development'
    ]
)


setup(**METADATA) 
