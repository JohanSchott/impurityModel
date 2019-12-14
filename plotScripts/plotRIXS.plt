#!/usr/bin/env gnuplot --persist -c

# Gnuplot plot script of RIXS spectra

# If no input argument is given, filename default is RIXS.bin
if (ARGC > 0) filename=ARG1; else filename='RIXS.bin'

# Set colormap
set palette rgb 23,28,3
set palette negative
#set pal gray

set xlabel 'w_{in}'
set ylabel 'w_{loss}'

# Plot the data in the binary file.
p filename binary matrix with image
