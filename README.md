# uGrid and uGridNET

uGrid is an open source code engineering design and decision tool developed to aid the engineering,
procurement and construction (EPC) of sustainable, reliable minigrids. This toolset is built by the
minigrid developer, OnePower Africa, to meet their development needs including resource sizing and 
distribution network layout design. The toolset optimizing for a minimum cost of electricity, referred
to as the levelized cost of electricity (LCOE). The toolset is composed of two tools, uGrid and
uGridNet.
<br/><br/>
uGrid performs resource sizing of solar, batteries, and propane generator based on
yearly load and weather profiles and a dispatch algorithm, all of which is customizable.
<br/><br/>
uGridNet
performs distribution network layout design based on electricity connection locations, generation
station location, exclusion ("can't build here") zones, and a reliability cost benefit assessment. 

## Getting Started

### Installation

1. Create a virtual environment using <a href='https://docs.python.org/3/library/venv.html'>`venv`<a/> or <a href="https://docs.python-guide.org/dev/virtualenvs/">`pipenv`<a/> for the project and activate it.
2. Install the requirements using `pip install -r requirements.txt` or `pipenv install`.

### Run

#### uGrid
The uGrid code has 3 python files: `technical_tools_PC_3_alt.py`, `economic_tools_PC_3.py`, and `macro_PC_3_alt.py` 
which contains the particle swarm algorithm. The program is run from `macro_PC_3_alt.py`, and it calls functions in the
`technical_tools_PC_3_alt.py` and `macro_PC_3_alt.py` files. 

All changes to adapt the code for a specific community are done from the input excel sheet called `uGrid_Input.xslx`. 
There are additional spreadsheets for weather and load that need to be within the same folder as the toolset.

See the Inputs section (in `uGrid Documentation.docx`) for information on what should be changed. Larger changes, such as technology changes, can be done in the code. See the sections on the python files for descriptions of the functions that can be changed for different control algorithms or technology changes. 

When the macro code is run the results from each generation from particle swarm optimization are outputted to the command line. An excel spreadsheet is outputted as the specified name in the Input spreadsheet. The output spreadsheet contains the information from each generation, and the global best results from the optimization. 

Functions exist in the technical python file to plot power flows. This is a manual process and can be improved by being automated in the code. 

To run the uGrid tool, change the inputs in `uGrid_Input.xlsx` and then run `python macro_PC_3_alt.py` in the terminal.
The results will be in the outputted excel spreadsheet. 


For more information, please read the `uGrid Documentation.docx` in this folder for information on
how to get started with these tools, including necessary input data and expected output.

## Authors

- Phylicia Cicilio (pcicilio@alaska.edu - University of Alaska Fairbanks, US)

- Matthew Orosz (mso@mit.edu - Massachusetts Institute of Technology, US)


## License

**The MIT License**

SPDX short identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


## Acknowledgments

The uGrid code is built from the works of Queralt Altes-Buch (qaltes@uliege.be - University of Liege, Energy Systems Research Unit, Belgium) 
and Matthew Orosz (mso@mit.edu - Massachusetts Institute of Technology, US)  - (github.com/queraltab/uGrid)
