Implementing NumPy (CPython) and Eigen(C++) and comparing
=========================================================
As of now the Spiesberger-Wahlberg 2002 localisation algorithm has
been implemented using both NumPy and Eigen. 

The Eigen code is being called using ```cppyy```. The overall speedup
because of using Eigen through ```cppyy``` is around 2-5X (depends on Unix/Windows). 
The average calculation time for a Numpy implementation is ~90ish micro seconds, 
while for the same Eigen run it is ~20ish micro seconds.