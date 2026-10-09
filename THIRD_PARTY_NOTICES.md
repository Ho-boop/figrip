# Third-party notices

## AM1.5G reference spectrum

`figrip/data/am15g.csv` contains the wavelength and global-tilt columns of the ASTM G173-03 reference spectrum. Its header records NREL as the scientific source and the pvlib-python dataset as the copied distribution source.

Scientific source: https://www.nrel.gov/grid/solar-resource/spectra-am1.5.html
Copied distribution: https://github.com/pvlib/pvlib-python/blob/main/pvlib/data/ASTMG173.csv
The following upstream BSD notice is retained for this copied distribution:
BSD 3-Clause License

Copyright (c) 2023 pvlib python Contributors
Copyright (c) 2014 PVLIB python Development Team
Copyright (c) 2013 Sandia National Laboratories

All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

  Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

  Redistributions in binary form must reproduce the above copyright notice, this
  list of conditions and the following disclaimer in the documentation and/or
  other materials provided with the distribution.

  Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Runtime dependencies

Figrip's own code is MIT-licensed. Dependencies are distributed under their own licenses; this repository does not replace those licenses. In particular, PyMuPDF/MuPDF has separate licensing terms: https://pymupdf.readthedocs.io/en/latest/about.html#license-and-copyright

NumPy, matplotlib, pytest and the optional OCR engine also retain their upstream licenses. No publisher PDF is bundled. The README preview is rendered from the project's synthetic model data.
