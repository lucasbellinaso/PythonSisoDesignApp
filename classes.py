import bokeh
from bokeh.models import ColumnDataSource, Span, Band, Label
from bokeh.plotting import figure as BkFig
from control.matlab import tf,c2d,bode,nyquist,rlocus,step,feedback,lsim
from control.matlab import margin, mag2db, db2mag
from scipy.signal import tf2zpk, zpk2tf
import numpy as np
from numpy import array, sin, cos, pi, exp, log, log10, sqrt, linspace, logspace
import ipywidgets as widgets
from ipywidgets import BoundedFloatText,Button,HBox,VBox,AppLayout,Dropdown
from IPython.display import display, clear_output
from matplotlib import pyplot as plt


class PoleOrZeroClass:
  """
    TYPE: 'pole', 'zero'
    SUBTYPE:'real', 'complex',  'integrator' or 'differentiator'
    Ts: None for continous time,  or float>0 for discrete-time
  """
  TYPE = 'pole'
  SUBTYPE = 'integrator'
  num, den = np.array([0,1]), np.array([0,1])
  freqHz, csi = 0, 1
  def __init__(self,TYPE,SUBTYPE,Ts,AppInstance,omega=None, csi=None):
    self.TYPE, self.SUBTYPE = TYPE, SUBTYPE
    self.Ts = Ts
    self.AppInstance = AppInstance
    self.ZPtf = tf(1,1,Ts)   #default: integrator pole

    box_layout = widgets.Layout(display='flex', align_items='stretch', width='200px')
    self.FrequencyWidget = BoundedFloatText(description=r"freq (Hz)",
                           value = (1. if Ts is None else 0.1/Ts ), min=0.001,
                           max=(1e6 if Ts is None else 0.499/Ts), step = 0.001,
                           continuous_update=True, layout = box_layout)
    if omega is not None: self.FrequencyWidget.value = omega/(2*pi)
    self.DampingRatioWidget = BoundedFloatText(description=r'Damp.Ratio',
                           value=(1. if csi is None else csi), step = 0.001,
                           min=0, max=1, continuous_update=True,layout = box_layout)
    self.FrequencySetButton = Button(description='Set',layout=widgets.Layout(width='100px'))
    self.DeleteButton = Button(description='Delete', layout=widgets.Layout(width='100px'))
    self.DeleteButton.on_click(self.deletePoleZero)
    self.FrequencySetButton.on_click(AppInstance.updateTFAndBokeh)
    self.PZIndexInApp = len(AppInstance.PolesAndZerosList)
    if SUBTYPE is 'real':
      self.PoleZeroDefineWidget = HBox([self.FrequencyWidget,self.FrequencySetButton])
      if self.Ts is None: self.num, self.den = array([0,1]),array([0,1])
      else:               self.num, self.den = array([0,1]),array([0,1])
    elif SUBTYPE is 'complex':
      if self.Ts is None: self.num,self.den = array([0,0,1]), array([0,0,1])
      else:               self.num,self.den = array([0,0,1]), array([0,0,1])
      self.PoleZeroDefineWidget = HBox([VBox([self.FrequencyWidget,
           self.DampingRatioWidget]), self.FrequencySetButton], Layout ='flex')     
    else: self.PoleZeroDefineWidget = widgets.Label('')
    Appwidget = AppLayout(header = None,
                      left_sidebar = widgets.Label(SUBTYPE+' '+TYPE),
                      center = self.PoleZeroDefineWidget,
                      right_sidebar = self.DeleteButton,
                      footer = None)
    self.Widget = Appwidget
    self.setFrequency(0)

  def __str__(self):
    return '{} {} {}'.format(self.TYPE, self.SUBTYPE, self.ZPtf)

  def setFrequency(self,b):
    if (self.freqHz == self.FrequencyWidget.value): 
         if (self.csi == self.DampingRatioWidget.value): return 0
    self.freqHz = self.FrequencyWidget.value
    self.csi = self.DampingRatioWidget.value
    poly, w0, csi = [] , 2*pi*self.freqHz, self.csi
    if self.Ts is None:  #continuous time system
      if   self.SUBTYPE is 'real':    poly = np.array([1/w0, 1])
      elif self.SUBTYPE is 'complex': poly = np.array([1/w0**2, 2*csi/w0, 1])
      else:                           poly = np.array([1,0])
    else:                #discrete-time system
      Ts, pz = self.Ts,  np.exp(-w0*self.Ts)
      if self.SUBTYPE is 'real':  poly = np.array([1,-pz])/(1-pz)
      elif self.SUBTYPE is 'complex':
            a1 = -2*exp(-self.csi*w0*Ts)*cos(w0*sqrt(1-csi**2)*Ts)
            a2 =  exp(-2*self.csi*w0*Ts)
            poly = array([1,a1,a2])/(1+a1+a2)
      else: poly = array([1/Ts,-1/Ts])
    if   self.TYPE is 'zero':  self.num = poly
    elif self.TYPE is 'pole':  self.den = poly
    self.ZPtf = tf(self.num,self.den,self.Ts)
    return 0

  def displayPZ(self):
    display(self.Widget)

  def printNumOrDen(self,num_den_key):
    if num_den_key is 'num' and self.TYPE is 'zero': poly = self.num
    elif  num_den_key is 'den' and self.TYPE is 'pole': poly = self.den
    else: return ''
    if self.Ts is None:
      if self.SUBTYPE is 'integrator' or self.SUBTYPE is 'differentiator': return 's'
    x = '*s' if self.Ts is None else '*z'
    str1 = f'({poly[0]:.4f}{x}'
    if self.SUBTYPE is 'complex': str1 = str1 + '²'
    if   poly[1]<0: str1 = str1+f'{poly[1]:.4f}'
    elif poly[1]>0: str1 = str1 + '+' + f'{poly[1]:.4f}'
    if self.SUBTYPE is 'complex':
      if   poly[2]<0:   str1 = str1+f'{x}{poly[2]:.4f}'
      elif poly[2]>0:   str1 = str1+f'{x}+{poly[2]:.4f}'
    return str1+')'
    
  def deletePoleZero(self,b):
    if self.TYPE is 'pole':
      if self.SUBTYPE is 'complex': self.AppInstance.relatOrderC -= 2
      else:                         self.AppInstance.relatOrderC -= 1
    elif self.SUBTYPE is 'complex': self.AppInstance.relatOrderC += 2
    else:                           self.AppInstance.relatOrderC += 1

    del self.AppInstance.PolesAndZerosList[self.PZIndexInApp]
    for x in range(self.PZIndexInApp,len(self.AppInstance.PolesAndZerosList)):
      self.AppInstance.PolesAndZerosList[x].PZIndexInApp -= 1 
    self.AppInstance.updateTFAndScreen(0)
  
 
 
class SISOApp:
  '''
  Class of SISO Design Python Application
      Control package:   import control.matlab as *
      GUI:  ipywidgets + Bokeh
  Initialization: 
      SISOApp(Gp, Gc, Gf)
             Gp: plant transfer function: continuous-time or discrete-time
             Gc (optional): controller transfer function
             Gf (optional - NOT IMPLEMENTED YET): sensor transfer function

        Control structure of a SISO system:
            r: reference
            u: controller output
            y: process output
            du, dy, dm:  input, output and measurement disturbances
                          du↓+         dy↓+
     r ──→+◯──→[Gc]──u───→+◯──→[Gp]──→+◯──┬──→ y
          -↑                         dm↓+    │
           └────────[Gf]←─────────────◯+←───┘ 

      Open Loop Transfer Function:  T(s) = Gc*Gp*Gf
  '''
  OShotIn = BoundedFloatText(value = 10, min=0, max=100, continuous_update=False,
                            layout = widgets.Layout(width='100px'))
  RTimeIn = BoundedFloatText(value = 0.1, min=0, continuous_update=False,
                             layout = widgets.Layout(width='100px'))
  STimeIn = BoundedFloatText(value = 0.1, min=0, continuous_update=False,
                             layout = widgets.Layout(width='100px'))
  ReqWidget = HBox([VBox([widgets.Label('Max Overshot (%)'), OShotIn  ])  ,
                   VBox([widgets.Label('Max Rise Time (s)'), RTimeIn  ]) ,
                   VBox([widgets.Label('Max Settling Time (s)'), STimeIn])] )
  NewPZDropdown = Dropdown(options=[' ','real pole','integrator','complex pole'],
                           value=' ', description=' ')
  CreatePZButton = Button(description='Insert and create figure below',
                          layout=widgets.Layout(width='200px'))
  NewPoleZeroBox = HBox([widgets.Label('Add Pole or Zero:'),NewPZDropdown,CreatePZButton])
  minCGainIndB, maxCGainIndB, CGainStepIndB = -80, 80, 0.5
  CgainInDBInteract = widgets.FloatSlider(value=0, min=minCGainIndB, 
                                       max=maxCGainIndB, step=CGainStepIndB, 
                                       layout=widgets.Layout(width='450px'),
                                       description = 'C gain dB:',
                                       continuous_update=True)
  updateControllerButton = Button(description='Update',
                          layout=widgets.Layout(width='200px'))  
  Appwidget = AppLayout(header = NewPoleZeroBox,
                        left_sidebar = widgets.Label('Poles and Zeros:'),
                        center = widgets.Label('Poles or Zero Widget'),
                        right_sidebar = updateControllerButton,
                        footer = HBox([CgainInDBInteract,ReqWidget]))
 
  dKgaindB = 0
  kvectLen = int((maxCGainIndB-minCGainIndB)/CGainStepIndB)+1
  rootsVect, kvect = [], list(np.linspace(minCGainIndB,maxCGainIndB,kvectLen))
  PhaseMargin, GainMargin = 0,0
  fNyquistHz = 1e6
  PolesAndZerosList = []
  CPoles, CZeros, Kgain, numC, denC = [], [], 1, np.array([0,1]), np.array([0,1])
  relatOrderC = 0
  OLTF = tf(1,1)


  def __init__(self, Gp, Gc=None, Gf=None):
    """Gp: plant transfer function (Python Control Package);
       Gc (optional): controller transfer function (Python Control Package)
       Gf (optional): measurement filter transfer function
       Gp, Gc e Gf shall be of the same sample time Ts.  """
    self.GpTransfFunc, self.Ts  = Gp, Gp.dt
    self.GpZeros, self.GpPoles,_ = tf2zpk(Gp.num[0][0], Gp.den[0][0])

    if self.Ts is not None:     self.fNyquistHz = 0.5/self.Ts;

    if Gc is None:
        self.CTransfFunc = tf(1,1, self.Ts)
    else:
        assert Gc.dt == Gp.dt ,  'Gc.dt should be equal of Gp.dt'
        self.CTransfFunc = Gc
    self.setControllerZPK(self.CTransfFunc)
    if Gf is None: self.GfTransfFunc = tf(1,1, self.Ts)
    else:
        assert Gf.dt == Gp.dt ,  'Gf.dt should be equal of Gp.dt'
        self.GfTransfFunc = Gf

    self.CreatePZButton.on_click(self.insertPoleZero)
    self.CgainInDBInteract.observe(self.updateGainAndBokeh,'value')
    self.updateControllerButton.on_click(self.updateTFAndBokeh)
    self.OShotIn.observe(self.updateRequirements,'value')
    self.RTimeIn.observe(self.updateRequirements,'value')
    self.STimeIn.observe(self.updateRequirements,'value')
    self.buildBokehFigs()
    self.createRLocus()
    self.updateCLabels()
    bokeh.io.output_notebook()
    self.updateTFAndScreen(0)

  def setControllerZPK(self, Gc):
    del self.PolesAndZerosList[:]
    self.numC, self.denC = Gc.num[0][0],Gc.den[0][0]
    self.CZeros, self.CPoles, self.Kgain  = tf2zpk(self.numC,self.denC)
    zeros_filt = list(filter(lambda x: x!=0, self.CZeros ))
    poles_filt = list(filter(lambda x: x!=0, self.CPoles ))
    num,den = zpk2tf(zeros_filt, poles_filt, 1)
    Gtemp = tf(num,den)
    self.Kgain = self.Kgain*Gtemp.dcgain()
    assert len(self.CPoles)>=len(self.CZeros), 'Gc should not have more zeros than poles.'
    self.CgainInDBInteract.value = 20*np.log10(self.Kgain)
    if Gc.dt is not None:
      assert all(np.abs(self.CPoles)<=1), 'Gc(z) should not have unstable poles.'
      assert all(np.abs(self.CZeros)<=1), 'Gc(z) should not have non minimum phase zeros.'
      omegaZ, omegaP = np.log(self.CZeros)/Gc.dt, np.log(self.CPoles)/Gc.dt
    else:
      assert all(np.real(self.CPoles)<=0), 'Gc(s) should not have unstable poles.'
      assert all(np.real(self.CZeros)<=0), 'Gc(s) should not have non minimum phase zeros.'
      omegaZ, omegaP = self.CZeros, self.CPoles

    for z in omegaZ:
      if z==0: self.PolesAndZerosList.append(
              PoleOrZeroClass('zero','differentiator', self.Ts, self))
      elif np.imag(z)>0:
        wn, csi = np.abs(z), np.abs(np.real(z))/np.abs(z)
        self.PolesAndZerosList.append(
              PoleOrZeroClass('zero','complex', self.Ts, self, omega=wn, csi=csi))
      elif np.imag(z)==0: self.PolesAndZerosList.append(
          PoleOrZeroClass('zero','real', self.Ts, self, omega=-z))
    for p in omegaP:
      if p==0:  self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','integrator', self.Ts, self))
      elif np.imag(p)>0: 
        wn, csi = np.abs(p), np.abs(np.real(p))/np.abs(p)
        self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','complex', self.Ts, self, omega=wn, csi=csi))
      elif np.imag(z)==0: self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','real', self.Ts, self,omega=np.abs(p)))

  def insertPoleZero(self,b):
    if self.NewPZDropdown.value is not ' ':
      PZtype_dict = {'integrator': ['pole','integrator'],
          'differentiator': ['zero','differentiator'],
          'real pole': ['pole','real'], 'real zero': ['zero','real'],
          'complex pole': ['pole','complex'],'complex zero': ['zero','complex']}
      PZtype, PZsubtype = PZtype_dict[self.NewPZDropdown.value]
      if PZtype is 'pole':
        if PZsubtype is 'complex': self.relatOrderC += 2
        else:                      self.relatOrderC += 1
      else:
        if PZsubtype is 'complex': self.relatOrderC -= 2
        else:                      self.relatOrderC -= 1
      self.PolesAndZerosList.append(PoleOrZeroClass(PZtype,PZsubtype,self.Ts,self))
      self.NewPZDropdown.value = ' '
      x = len(self.PolesAndZerosList)-1
      self.PolesAndZerosList[x].FrequencySetButton.on_click(self.updateTFAndBokeh)
      self.updateTFAndScreen(0)

  def printController(self,b):
    numstr, denstr = '', ''
    Kstr = f'{db2mag(self.CgainInDBInteract.value):.2f}'
    for pz in self.PolesAndZerosList:
      numstr = numstr + pz.printNumOrDen('num')
      denstr = denstr + pz.printNumOrDen('den')
    if numstr is '': numstr = '1'
    if denstr is '': denstr = '1'
    #print('C(s)= {}*{}/({})'.format(Kstr,numstr, denstr))
    return numstr, denstr, Kstr
    
  def buildBokehFigs(self):
    #BOKEH FIGURES:
    #Vector data:
    self.bodesource = ColumnDataSource( data={'omega':[], 'freqHz': [],
              'magdBT':[],'magT':[],'magdBG':[],'magG': [],'angT':[],'angG':[]})
    self.gpbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.gzbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.cpbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.czbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.PM_GMsource = ColumnDataSource( data = {'PMfcHz': [1.,1.],'GMfHz':[2.,2.],
                                 'ylimsmag':[-200,200], 'ylimsang':[-720,720] })
    self.rlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.gprlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.gzrlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.cprlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.czrlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.krlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.stepsource = ColumnDataSource(
                data={'t_s':[],'stepRYmf':[],'stepUYma':[],'stepRUmf':[]})
  
    #Shadows:
    MAX_OVERSHOT = 0.01*self.OShotIn.value + 1
    MAX_RISE_TIME, MAX_SETTLING_TIME = self.RTimeIn.value, self.STimeIn.value
    _thetaZ = np.linspace(0,np.pi,100)
    _costh, _sinthN, _sinth = np.cos(_thetaZ), -np.sin(_thetaZ), np.sin(_thetaZ)
    self.shadowsource = ColumnDataSource(
        data={'x_s': [0,1e4],     'ylow': [-1e4,1e4],  'yup': [1e4,1e4],  
            'xn_z': [-1e4,-1], 'xp_z': [1,1e4] , 'zero':[0,0],
            'overshot':[MAX_OVERSHOT, MAX_OVERSHOT], 
            'risetime':[MAX_RISE_TIME,1e4] , 'riselevel':[0.9,0.9],
            'settlingtime':[MAX_SETTLING_TIME,1e4],
            'setlevel1':[0.98,0.98], 'setlevel2':[1.02,1.02]  } )
    self.shadowZsource=ColumnDataSource(
               data = {'x_z':_costh, 'ylow_z':_sinthN, 'yup_z':_sinth, 
                       'ylow': 100*[-1e4], 'yup': 100*[1e4]})

    self.shadows = {
    'rloc_s': Band(base='x_s', lower='ylow', upper='yup', level='underlay',
            source=self.shadowsource, fill_color='lightgrey', line_color='black'),
    'rloc_z1': Band(base='xn_z', lower='ylow', upper='yup', level='underlay', 
            source=self.shadowsource, fill_color='lightgrey'),
    'rloc_z2': Band(base='xp_z', lower='ylow', upper='yup', level='underlay', 
            source=self.shadowsource, fill_color='lightgrey'),
    'rloc_z3': Band(base='x_z', lower='ylow', upper='ylow_z', level='underlay',
           source=self.shadowZsource,fill_color='lightgrey',line_color='black'),
    'rloc_z4': Band(base='x_z', lower='yup_z', upper='yup', level='underlay',
           source=self.shadowZsource,fill_color='lightgrey',line_color='black'),
    'ovsht': Band(base='x_s', lower='overshot', upper='yup', level='underlay',
            source=self.shadowsource,fill_color='lightgrey', visible=True),
    'riset': Band(base='risetime', lower='ylow', upper='riselevel', 
            level='underlay', source=self.shadowsource,fill_color='lightgrey'),
    'sett1': Band(base='settlingtime', lower='riselevel', upper='setlevel1', 
                 level='underlay', source=self.shadowsource,fill_color='lightgrey'),
    'sett2': Band(base='settlingtime', lower='setlevel2', upper='overshot', 
                 level='underlay', source=self.shadowsource,fill_color='lightgrey') }                    

    _TTS_BD1 = [('sys',"$name"),("f","$x Hz"),("mag","$y dB")]
    _TTS_BD2 = [('sys',"$name"),("f","$x Hz"),("ang","$y°")]
    _TTS_RLOC= [("real","@x"),("imag","@y"),('K','@K{0.0 a}')]
    _TTS_TRESP = [('signal', "$name"), ("t", "$x s"), ("value", "$y") ]
    self.figMag = BkFig(title="Bode Magnitude", plot_height=300, plot_width=400,
               toolbar_location="above", tooltips = _TTS_BD1, x_axis_type="log",
               x_axis_label='f (Hz)', y_axis_label='mag (dB)')
    self.figAng =  BkFig(title="Bode Angle", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_BD2, x_axis_type="log",
                x_axis_label='f (Hz)', y_axis_label='ang (°)')
    self.figAng.x_range = self.figMag.x_range   #same axis
    self.figAng.yaxis.ticker=np.linspace(-720,720,17)
    self.figRLoc=  BkFig(title="Root Locus", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_RLOC,
                x_axis_label='real', y_axis_label='imag')
    self.figTResp = BkFig(title="Time Response", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_TRESP,
                x_axis_label='time (s)', y_axis_label='y') 
    self.figTResp2= BkFig(title="Time Response", plot_height=300, plot_width=800, 
                toolbar_location="above", tooltips = _TTS_TRESP,
                x_axis_label='time (s)', y_axis_label='y')
 
    self.Bkgrid = bokeh.layouts.layout([[self.figMag, self.figRLoc],
                                        [self.figAng, self.figTResp]])


    if self.Ts is None:   #continuous time
      self.figRLoc.add_layout(self.shadows['rloc_s'])
    else:                 #discrete time
        for strkey in ['rloc_z1', 'rloc_z2', 'rloc_z3', 'rloc_z4']:
          self.figRLoc.add_layout(self.shadows[strkey])
        self.Nyquistlimits = Span(location=0.5/self.Ts,
                                 dimension='height', line_color='black',
                                 line_dash='dotted', line_width=1)
        self.figMag.add_layout(self.Nyquistlimits)
        self.figAng.add_layout(self.Nyquistlimits)
    for strkey in ['ovsht', 'riset', 'sett1', 'sett2']:
      self.figTResp.add_layout(self.shadows[strkey])


    #Bode Diagram:
    bodemagT=self.figMag.line(x='freqHz', y='magdBT',color="blue",line_width=1.5,
                alpha=0.8,name='|T(s)|',legend_label='T(s)',source=self.bodesource)
    bodemagG=self.figMag.line(x='freqHz', y='magdBG',color="green",line_width=1.5, 
                alpha=0.8,name='|Gp(s)|',line_dash='dashed',
                legend_label='Gp(s)',source=self.bodesource)
    bodeangT=self.figAng.line(x='freqHz', y='angT', color="blue", line_width=1.5,
                alpha=0.8, name='∡T(s)', source=self.bodesource)
    bodeangG=self.figAng.line(x='freqHz', y='angG',color="green", line_width=1.5,
                alpha=0.8,name='∡Gp(s)',line_dash='dashed',source=self.bodesource)
    bodeGpmag = self.figMag.x( x='fHz',y='magdB',line_color='blue', size=8,
                 name='Gp poles', source = self.gpbodesource)
    bodeGpang=self.figAng.x(x='fHz',y='angdeg',line_color='blue', size=8,
                 name='Gp poles angle', source = self.gpbodesource)
    bodeGzmag = self.figMag.circle(x='fHz',y='magdB',line_color='blue',size=6,
                 name='Gp zeros',fill_color=None, source = self.gzbodesource)
    bodeGzang=self.figAng.circle(x='fHz',y='angdeg',line_color='blue',size=6,
              name='Gp zeros angle', fill_color=None,source = self.gzbodesource)
    bodeCpmag = self.figMag.x(x='fHz',y='magdB',line_color='red',size=8,
                 name='C poles', source = self.cpbodesource)
    bodeCpang=self.figAng.x(x='fHz',y='angdeg',line_color='red',size=8,
                 name='C poles angle', source = self.cpbodesource)
    bodeCzmag = self.figMag.circle(x='fHz',y='magdB',line_color='red',size=6,
                 name='C zeros', fill_color=None, source = self.czbodesource)
    bodeCzang=self.figAng.circle(x='fHz',y='angdeg',line_color='red',size=6,
                name='C zeros angle',fill_color=None, source = self.czbodesource)
    self.GMSpan = Span(location=1, dimension='height',
                       line_color='black', line_dash='dotted', line_width=1)
    self.PMSpan = Span(location=1, dimension='height',
                       line_color='black', line_dash='dotted', line_width=1)
    self.PMtxt = Label(x=5, y=5, x_units='screen', y_units='screen', 
                         text=' ',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.GMtxt = Label(x=5, y=20, x_units='screen', y_units='screen', 
                         text=' ',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Clbltxt = Label(x=5, y=20, x_units='screen', y_units='screen', 
                         text=' ',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Clbltxt.text = 'C(s) = ' if self.Ts is None else 'C(z) = '
    self.Cgaintxt = Label(x=40, y=20, x_units='screen', y_units='screen', 
                         text='K',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Cnumtxt = Label(x=100, y=30, x_units='screen', y_units='screen', 
                         text='num',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Cdentxt = Label(x=100, y=10, x_units='screen', y_units='screen', 
                         text='den',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.figMag.add_layout(self.GMSpan), self.figAng.add_layout(self.GMSpan)
    self.figMag.add_layout(self.PMSpan), self.figAng.add_layout(self.PMSpan)
    self.figAng.add_layout(self.PMtxt), self.figAng.add_layout(self.GMtxt)
    self.figMag.add_layout(self.Clbltxt), self.figMag.add_layout(self.Cgaintxt)
    self.figMag.add_layout(self.Cnumtxt), self.figMag.add_layout(self.Cdentxt)

    #Root Locus:
    rlocusline = self.figRLoc.dot(x='x',y='y',color='blue',
                                  name='rlocus', source = self.rlocussource)
    rlocusGpoles = self.figRLoc.x(x='x',y='y',color='blue', size=8,
                                name='Gp pole', source = self.gprlocussource)
    rlocusGzeros = self.figRLoc.circle(x='x',y='y',line_color='blue',size=6,
                 name='Gp zero', fill_color=None, source = self.gzrlocussource)
    rlocusCpoles = self.figRLoc.x(x='x',y='y',color='red', size=8,
                                name='C pole', source = self.cprlocussource)
    rlocusCzeros = self.figRLoc.circle(x='x',y='y',line_color='red',size=6,
                 name='C zero', fill_color=None, source = self.czrlocussource)
    rlocusMF = self.figRLoc.square(x='x',y='y', line_color='red',size=6,
                 name='K', fill_color='red', source = self.krlocussource)    
    self.Stabilitytxt = Label(x=10, y=200, x_units='screen', y_units='screen', 
                         text=' ',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.figRLoc.add_layout(self.Stabilitytxt)

    #Step response:
    self.figTResp.extra_y_ranges = {'u_range': bokeh.models.Range1d()}
    self.figTResp.add_layout(bokeh.models.LinearAxis(y_range_name="u_range",
                                                     axis_label='u'), 'right')
    self.figTResp.y_range = bokeh.models.Range1d(start = -0.1, end = 1.4)
    if self.Ts is None:
      stepR2Y=self.figTResp.line(x='t_s', y='stepRYmf',color="blue",
                                 line_width=1.5, name='y',
                        legend_label='y (closed loop)',  source=self.stepsource)
      stepU2Y=self.figTResp.line(x='t_s', y='stepUYma',color="green",
               legend_label='y (open loop)', line_dash='dashed', line_width=1.0,
                                name='y (ol)',source=self.stepsource)
      stepR2U=self.figTResp.line(x='t_s', y='stepRUmf',color="red",
                       line_width=1.0, name='u',legend_label='u (closed loop)',
            line_dash='dashed', source=self.stepsource, y_range_name = 'u_range')
    else:
      stepR2Y=self.figTResp.dot(x='t_s', y='stepRYmf',color="blue",
                                 line_width=1.5, name='y',  size=15,
                        legend_label='y (closed loop)',  source=self.stepsource)
      stepU2Y=self.figTResp.dot(x='t_s', y='stepUYma',color="green", size=15,
                                legend_label='y (open loop)', line_width=1.0,
                                name='y (ol)',source=self.stepsource)
      stepR2U=self.figTResp.dot(x='t_s', y='stepRUmf',color="red", size=15,
                       line_width=1.0, name='u',legend_label='u (closed loop)',
                       source=self.stepsource, y_range_name = 'u_range')
    self.figTResp.legend.location = 'bottom_right'
    self.figTResp.legend.click_policy = 'hide'

    #self.createBode()

  def updateScreen(self):
    #clear_output()  #not working
    npz = [' ','real pole','integrator','complex pole']
    if (self.relatOrderC >= 1): npz.extend(['real zero','differentiator'])
    if (self.relatOrderC >= 2): npz.append('complex zero')
    self.NewPZDropdown.options = npz
   
    self.Appwidget.center = VBox([pzs.Widget for pzs in self.PolesAndZerosList])
    asdfs = display(self.Appwidget)
    self.printController(0)
    self.createRLocus()
    self.createBode()
    self.updateStepResponse()
    self.Bknb_handle = bokeh.io.show(self.Bkgrid, notebook_handle=True)
    bokeh.io.push_notebook(handle = self.Bknb_handle)

  def updateGainAndBokeh(self,b):
    #Update gain:  updates Gc(s),  T(s), and Kgain
    Kgain_new = db2mag(self.CgainInDBInteract.value)
    dKgain = Kgain_new/self.Kgain
    self.CTransfFunc = self.CTransfFunc*dKgain
    self.OLTF = self.OLTF*dKgain
    self.Kgain, self.dKgaindB = Kgain_new,  np.round(mag2db(dKgain),decimals=1)
    #Update Bokeh:
    self.updateBokeh()

  def updateTFAndBokeh(self,b):
    self.updateTransferFunction()
    self.createBode()
    self.createRLocus()
    self.updateBokeh()

  def updateTFAndScreen(self,b):
    self.updateTransferFunction()
    self.updateScreen()

  def updateBokeh(self):
    self.updateBodeData()
    self.updateRLocusData()
    self.updateStepResponse()
    self.updateCLabels()
    bokeh.io.push_notebook(handle = self.Bknb_handle)
  
  def updateTransferFunction(self):
    self.Kgain = db2mag(self.CgainInDBInteract.value)
    self.numC, self.denC = np.array([0,self.Kgain]) ,  np.array([0,1])
    for pz in self.PolesAndZerosList:
      pz.setFrequency(0)
      self.numC = np.polymul(self.numC, pz.num)
      self.denC = np.polymul(self.denC, pz.den)
    self.CTransfFunc = tf(self.numC, self.denC, self.Ts)
    self.CPoles = self.CTransfFunc.pole()
    self.CZeros = self.CTransfFunc.zero()
    #print(self.CPoles)
    self.OLTF = self.GpTransfFunc*self.CTransfFunc

  def createBode(self):
    '''Creates the plots for Bode Diagram '''
    magT,phiT,omega = bode(self.OLTF, Plot=False)
    magG,phiG,_ = bode(self.GpTransfFunc,omega, Plot=False)
    magdbG, magdbT = mag2db(magG), mag2db(magT)
    phiGHz, phiTHz = phiG*180/pi, phiT*180/pi
    self.bodesource.data={'omega':omega, 'freqHz':(omega/(2*np.pi)),
                         'magdBG':magdbG, 'magG':magG, 'angG':phiGHz,
                         'magdBT':magdbT, 'magT':magT, 'angT':phiTHz}
    self.updatePMGM()
    def d2c_clampAtNyquistFreq(PZdiscr):
          pzabs = np.abs(PZdiscr)
          pcont = np.abs(np.log(pzabs))/self.Ts
          omega_nyqu = 2*np.pi*self.fNyquistHz
          for x in pcont:
             if x>omega_nyqu: x = omega_nyqu
          return pcont
    func1 = np.abs if self.Ts is None else d2c_clampAtNyquistFreq
    dict1 = {'Gpp': [self.GpPoles,self.gpbodesource],
             'Gpz': [self.GpZeros,self.gzbodesource],
             'CP' : [self.CPoles,self.cpbodesource],
             'CZ' : [self.CZeros,self.czbodesource]}
    for key1 in ['Gpp','Gpz','CP','CZ']:
      pORz = list(filter(lambda x: x>1e-10, func1(dict1[key1][0])))
      if pORz:
        mag,phi,omega = bode(self.OLTF, pORz, Plot=False)
        magdB, phideg, fHz = mag2db(mag), phi*180/pi, (omega/(2*np.pi))
        dict1[key1][1].data={'fHz':list(fHz),'magdB':list(magdB),
                           'angdeg':list(phideg)}

  def updateBodeData(self):
    def sum_constant_to_list(data_dict, list_key, constant):
          data_dict[list_key] = list(np.array(data_dict[list_key])+constant)
    dmagdB, dmag = self.dKgaindB, db2mag(self.dKgaindB)
    for pz in [self.gpbodesource,self.gzbodesource,self.cpbodesource,self.czbodesource]:
        sum_constant_to_list(pz.data,'magdB', dmagdB)
    sum_constant_to_list(self.bodesource.data,'magdBT', dmagdB )
    self.bodesource.data['magT']=list(dmag*np.array(self.bodesource.data['magT']))
    self.updatePMGM()

  def updatePMGM(self):
    self.GainMargin,self.PhaseMargin,wg,wc = margin(self.OLTF)
    if np.isnan(wg): wg = 2*np.pi*self.fNyquistHz
    if np.isnan(wc): wc = 2*np.pi*self.fNyquistHz
    self.PMSpan.location = wc/(2*np.pi)
    self.GMSpan.location = wg/(2*np.pi)
    if str(self.GainMargin) is 'inf':  self.GMtxt.text = 'GM: inf'
    else: self.GMtxt.text = f'GM:{self.GainMargin:.1f} dB'
    if str(self.PhaseMargin) is 'inf': self.PMtxt.text = 'PM: inf'
    else: self.PMtxt.text = f'PM: {self.PhaseMargin:.1f}°'

  def createRLocus(self):
    CgaindB = self.CgainInDBInteract.value
    Cgain = db2mag(CgaindB)
    self.kvectLen= int((self.maxCGainIndB-self.minCGainIndB)/self.CGainStepIndB+1)
    kvectdB = np.linspace(self.minCGainIndB, self.maxCGainIndB, self.kvectLen)
    self.kvect = db2mag(kvectdB)
    Kgp, Kgz = np.zeros(len(self.GpPoles)), np.Inf*np.ones(len(self.GpZeros))
    Kcp, Kcz = np.zeros(len(self.CPoles)), np.Inf*np.ones(len(self.CZeros))
    self.rootsVect,_ = rlocus(self.OLTF/Cgain,Plot=False,kvect=self.kvect)
    re, im, rootsVect = np.real, np.imag, self.rootsVect
    Krlocus,  cols = self.kvect,  self.rootsVect.shape[1]-1
    for x in range(cols):  Krlocus = np.column_stack((Krlocus,self.kvect))
    self.rlocussource.data = {'x':re(rootsVect),'y':im(rootsVect),'K':Krlocus}
    self.gprlocussource.data = {'x':re(self.GpPoles),'y':im(self.GpPoles),'K':Kgp}
    self.gzrlocussource.data = {'x':re(self.GpZeros),'y':im(self.GpZeros),'K':Kgz}
    self.cprlocussource.data = {'x':re(self.CPoles),'y':im(self.CPoles),'K':Kcp}
    self.czrlocussource.data = {'x':re(self.CZeros),'y':im(self.CZeros),'K':Kcz}
    self.updateRLocusData()

  def updateRLocusData(self):
    Cgain_real = db2mag(self.CgainInDBInteract.value)
    Kindex = int(self.kvectLen*(self.CgainInDBInteract.value-self.minCGainIndB)
                                       /(self.maxCGainIndB-self.minCGainIndB))
    x,y = np.real(self.rootsVect[Kindex]), np.imag(self.rootsVect[Kindex])
    K = self.kvect[Kindex]*np.ones(len(list(x)))
    self.krlocussource.data = {'x': x , 'y': y, 'K':K} 
    comp = (x>0) if self.Ts is None else  ((x*x+y*y)>1)
    if any(comp): self.Stabilitytxt.text = 'Unstable Loop'
    else:         self.Stabilitytxt.text = 'Stable Loop'

  def updateStepResponse(self):
    Gmf = feedback(self.OLTF, 1)
    Gru = feedback(self.CTransfFunc, self.GpTransfFunc)
    ymf,tvec = step(Gmf)
    #rt = tvec[next(i for i in range(0,len(ymf)-1) if ymf[i]>ymf[-1]*.90)]-tvec[0]
    try: st = tvec[next(len(ymf)-q for q in range(2,len(ymf)-1) if abs(ymf[-q]/ymf[-1])>1.02)]
    except: st = 0.33*tvec[-1]
    if self.Ts is None:
        ymf,tvec = step(Gmf, T=np.linspace(0,3*st,100))
    yma,_ = step(self.GpTransfFunc, T=tvec)
    umf,_ = step(Gru, T=tvec)
    self.stepsource.data={'t_s':tvec,'stepRYmf':ymf,'stepUYma':yma,'stepRUmf':umf }
    self.figTResp.extra_y_ranges['u_range'].update(start=0.9*np.min(umf),
                                                   end=1.2*np.max(umf))

  def updateRequirements(self,b):
    max_overshot = 0.01*self.OShotIn.value+1
    self.shadowsource.data['overshot'] = [max_overshot, max_overshot]
    self.shadowsource.data['risetime'] = [self.RTimeIn.value, 1e4]
    self.shadowsource.data['settlingtime'] = [self.STimeIn.value, 1e4]
    bokeh.io.push_notebook(handle = self.Bknb_handle)

  def updateCLabels(self):
    self.Cnumtxt.text,self.Cdentxt.text,self.Cgaintxt.text = self.printController(0)
