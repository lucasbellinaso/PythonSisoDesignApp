# Python Siso Design App

User interface for designing Single-Input-Single-Output Controllers in Google Colab.

Developer: Prof. Lucas Vizzotto Bellinaso
Universidade Federal de Santa Maria

## Copy this code to the first cell in Google Colab:

``` python
# Use in the first cell:
!pip install bokeh             # Bokeh package must be installed in Colab server
!pip install control           # Control package must be installed in Colab server
!git clone https://github.com/lucasbellinaso/PythonSisoDesignApp.git
%cd PythonSisoDesignApp        # browsing the github folder
!python classes.py             # running the github code
from classes import SISOApp    # importing the SISOApp
from control.matlab import *   # importing control package as a Matlab environment
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
