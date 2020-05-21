# uGrid

uGrid is an open source code engineering design and decision tool developed to aid the engineering,
procurement and construction (EPC) of sustainable, reliable minigrids. This toolset is built by the
minigrid developer, OnePower Africa, to meet their development needs including resource sizing and 
distribution network layout design. The toolset optimizing for a minimum cost of electricity, referred
to as the levelized cost of electricity (LCOE). The toolset is composed of two tools, uGrid and
uGridNet. uGrid performs resource sizing of solar, batteries, and propane generator based on
yearly load and weather profiles and a dispatch algorithm, all of which is customizable. uGridNet
performs distribution network layout design based on electricity connection locations, generation
station location, exclusion ("can't build here") zones, and a reliability cost benefit assessment. 

## Getting Started
Please read the "uGrid Documentation" word document in this folder for information on
how to get started with these tools, including necessary input data and expected output.

## Prerequisites
The user will need Python 3 installed and the python packages: numpy, pandas, math, pdf2image,
convert_from_path, PIL, time, and matplotlib.

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
