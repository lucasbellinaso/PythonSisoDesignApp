# Python Siso Design App

User interface for designing Single-Input-Single-Output Controllers in Google Colab.

Developer: Prof. Lucas Vizzotto Bellinaso
Universidade Federal de Santa Maria

## Copy this code to the first cell in Google Colab:

``` python
# Use in the first cell:
!pip install bokeh
!pip install control
from control.matlab import *
!git clone https://github.com/lucasbellinaso/PythonSisoDesignApp.git
%cd PythonSisoDesignApp
!python classes.py
from classes import SISOApp
```


## To see help:

``` python
help(SISOApp)
```

## Example

``` python
Gc = tf(1,[1,1,1])  #example plant
SISOApp(Gc)
```
