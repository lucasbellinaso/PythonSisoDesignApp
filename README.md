# Python Siso Design App

User interface for designing Single-Input-Single-Output Controllers in Google Colab.

Developer: Prof. Lucas Vizzotto Bellinaso
Universidade Federal de Santa Maria

To run in Colab:

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


# To verify how to use the application:

``` python
help(SISOApp)
```

# Example how to use the application to design a controller
``` python
Gc = tf(1,[1,1,1])  #example plant
SISOApp(Gc)
```
