#!/bin/bash

for i in figure_*.pdf; do
  base=`echo $i | sed 's/\.pdf//'`
  gs -dUseTrimBox -sDEVICE=pngalpha -dBATCH -dNOPAUSE -r180x180 -sOutputFile=${base}-%d.png ${base}.pdf
done
