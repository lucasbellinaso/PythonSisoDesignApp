import bokeh
from bokeh.models import ColumnDataSource, Span, Band, Label
from bokeh.plotting import figure as BkFig
from control.matlab import tf,c2d,bode,nyquist,rlocus,step,feedback,lsim,minreal
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
  def __init__(self,TYPE,SUBTYPE,Ts,AppInstance,omega=None, csi=None):
    self.TYPE, self.SUBTYPE = TYPE, SUBTYPE
    self.num, self.den = np.array([0,1]), np.array([0,1])
    self.freqHz, self.csi = 0, 1
    self.Ts = Ts
    self.AppInstance = AppInstance
    self.ZPtf = tf(1,1,Ts)   #default: integrator pole

    box_layout = widgets.Layout(display='flex', align_items='stretch', width='200px')
    self.FrequencyWidget = BoundedFloatText(description=r"freq (Hz)",
                           value = (1. if Ts in [None, 0.0] else 0.1/Ts ), min=0.001,
                           max=(1e6 if Ts in [None, 0.0] else 0.499/Ts), step = 0.001,
                           continuous_update=True, layout = box_layout)
    if omega != None: self.FrequencyWidget.value = omega/(2*pi)
    self.DampingRatioWidget = BoundedFloatText(description=r'Damp.Ratio',
                           value=(0.1 if csi == None else csi), step = 0.001,
                           min=0, max=1, continuous_update=True,layout = box_layout)
    self.FrequencySetButton = Button(description='Set',layout=widgets.Layout(width='100px'))
    self.DeleteButton = Button(description='Delete', layout=widgets.Layout(width='100px'))
    self.DeleteButton.on_click(self.deletePoleZero)
    self.FrequencySetButton.on_click(AppInstance.updateTFAndBokeh)
    self.PZIndexInApp = len(AppInstance.PolesAndZerosList)
    if SUBTYPE == 'real':
      self.PoleZeroDefineWidget = HBox([self.FrequencyWidget,self.FrequencySetButton])
      if self.Ts in [None, 0.]: self.num, self.den = array([0,1]),array([0,1])
      else:               self.num, self.den = array([0,1]),array([0,1])
    elif SUBTYPE == 'complex':
      if self.Ts in [None, 0.0]: self.num,self.den = array([0,0,1]), array([0,0,1])
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
    if self.Ts in [None, 0.0]:  #continuous time system
      if   self.SUBTYPE == 'real':    poly = np.array([1/w0, 1])
      elif self.SUBTYPE == 'complex': poly = np.array([1/w0**2, 2*csi/w0, 1])
      else:                           poly = np.array([1,0])
    else:                #discrete-time system
      Ts, pz = self.Ts,  np.exp(-w0*self.Ts)
      if self.SUBTYPE == 'real':  poly = np.array([1,-pz])/(1-pz)
      elif self.SUBTYPE == 'complex':
            a1 = -2*exp(-self.csi*w0*Ts)*cos(w0*sqrt(1-csi**2)*Ts)
            a2 =  exp(-2*self.csi*w0*Ts)
            poly = array([1,a1,a2])/(1+a1+a2)
      else: poly = 1/Ts*array([1,-1])  #integrator or differentiator
    if   self.TYPE == 'zero':  self.num = poly
    elif self.TYPE == 'pole':  self.den = poly
    self.ZPtf = tf(self.num,self.den,self.Ts)
    return 0

  def displayPZ(self):
    display(self.Widget)

  def printNumOrDen(self,num_den_key):
    if num_den_key == 'num' and self.TYPE == 'zero': poly = self.num
    elif  num_den_key == 'den' and self.TYPE == 'pole': poly = self.den
    else: return ''
    if self.SUBTYPE in ['integrator','differentiator']:
      return 's' if self.Ts in [None, 0.0] else f'{poly[0]:.2f}(z-1)'
    if self.SUBTYPE == 'real':
      if self.Ts in [None, 0.0]: return f'(s/{(1/poly[0]):.2f}+1)'
      else: return f'{poly[0]:.2f}(z{(poly[1]/poly[0]):.4f})'
    if self.SUBTYPE == 'complex':
      if self.Ts in [None, 0.0]:
        w0 = (2*pi*self.freqHz)
        a1s = '' if self.csi == 0  else f'+s(2*{self.csi:.3f}/{w0:.2f})'
        return f'((s/{w0:.2f})²{a1s}+1)'
      else:
        az0 = '+1))' if (self.csi==0) else f'+{(poly[2]/poly[0]):.4f}))'
        signal1 = '+' if (poly[1]>0) else ''
        return f'({poly[0]:.4f}(z²{signal1}{(poly[1]/poly[0]):.4f}z'+az0

    
  def deletePoleZero(self,b):
    # Changes the relative order when pole or zero is deleted:
    delta = (-1 if self.TYPE == 'pole' else 1)*(2 if self.SUBTYPE == 'complex' else 1)
    self.AppInstance.relatOrderC += delta
    if self.AppInstance.relatOrderC<0:
      print('Controller should not have more zeros than poles. First delete a zero!')
      self.AppInstance.relatOrderC -= delta
    else:
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
  
  minCGainIndB, maxCGainIndB, CGainStepIndB = -80, 80, 0.5
  kvectLen = int((maxCGainIndB-minCGainIndB)/CGainStepIndB)+1
  kvect = list(np.linspace(minCGainIndB,maxCGainIndB,kvectLen))


  def __init__(self, Gp, Gc=None, Gf=None):
    """Gp: plant transfer function (Python Control Package);
       Gc (optional): controller transfer function (Python Control Package)
       Gf (optional): measurement filter transfer function
       Gp, Gc e Gf shall be of the same sample time Ts.  """
    clear_output()
    self.GpTransfFunc, self.Ts  = Gp, Gp.dt
    self.GpZeros, self.GpPoles,_ = tf2zpk(Gp.num[0][0], Gp.den[0][0])
    self.PolesAndZerosList, self.relatOrderC = [], 0
    self.CPoles,self.CZeros,self.Kgain, self.dKgaindB = [], [], 1, 0
    self.numC, self.denC = np.array([0,1]), np.array([0,1])
    self.OLTF = tf(1,1,self.Ts)
    self.PhaseMargin, self.GainMargin = 0,0
    self.rootsVect = []
    self.fNyquistHz = 1e6 if self.Ts in [None, 0.0] else 0.5/self.Ts;
    if Gc == None:
        self.CTransfFunc = tf(1,1, self.Ts)
    else:
        condicoes = [Gp.dt in [None, 0.0] and Gp.dt in [None, 0.0], Gc.dt==Gp.dt]
        assert any(condicoes) ,  'Gc.dt should be equal to Gp.dt'
        self.CTransfFunc = Gc
    if Gf == None: self.GfTransfFunc = tf(1,1, self.Ts)
    else:
        condicoes = [Gp.dt in [None, 0.0] and Gf.dt in [None, 0.0], Gf.dt==Gp.dt]
        assert any(condicoes) ,  'Gf.dt should be equal to Gp.dt'
        self.GfTransfFunc = Gf

    #Create ipywidgets layout and events:
    self.CreatePZButton = Button(description='Insert and create figure below',
                          layout=widgets.Layout(width='200px'))
    self.NewPoleZeroBox = HBox([widgets.Label('Add Pole or Zero:'),
                                self.NewPZDropdown,self.CreatePZButton])
    self.CgainInDBInteract = widgets.FloatSlider(value=0, min=self.minCGainIndB, 
                                       max=self.maxCGainIndB, step=self.CGainStepIndB, 
                                       layout=widgets.Layout(width='450px'),
                                       description = 'C gain dB:',
                                       continuous_update=True)    
    self.updateControllerButton = Button(description='Update and print controller',
                          layout=widgets.Layout(width='200px')) 
    self.Appwidget = AppLayout(header = self.NewPoleZeroBox,
                        left_sidebar = widgets.Label('Poles and Zeros:'),
                        center = widgets.Label('Poles or Zero Widget'),
                        right_sidebar = self.updateControllerButton,
                        footer = HBox([self.CgainInDBInteract,self.ReqWidget]))
    self.setControllerZPK(self.CTransfFunc)
    self.buildBokehFigs()
    self.updateTFAndScreen(0)

    #Define events:
    self.CreatePZButton.on_click(self.insertPoleZero)
    self.CgainInDBInteract.observe(self.updateGainAndBokeh,'value')
    self.updateControllerButton.on_click(self.updateAndPrintC)
    self.OShotIn.observe(self.updateRequirements,'value')
    self.RTimeIn.observe(self.updateRequirements,'value')
    self.STimeIn.observe(self.updateRequirements,'value')
    #bokeh.io.output_notebook()

  def setControllerZPK(self, Gc):
    del self.PolesAndZerosList[:]
    self.numC, self.denC = Gc.num[0][0],Gc.den[0][0]
    self.CZeros, self.CPoles, self.Kgain  = tf2zpk(self.numC,self.denC)
    self.relatOrderC = len(self.CPoles) - len(self.CZeros)
    pzIntDiff = 0.0 if self.Ts in [None, 0.0] else 1.0
    zeros_filt = list(filter(lambda x: np.abs(x-pzIntDiff)>=1e-6, self.CZeros ))
    poles_filt = list(filter(lambda x: np.abs(x-pzIntDiff)>=1e-6, self.CPoles ))
    num,den = zpk2tf(zeros_filt, poles_filt, 1)
    Gtemp = tf(num,den, self.Ts)
    self.Kgain = Gtemp.dcgain()
    assert len(self.CPoles)>=len(self.CZeros), 'Gc should not have more zeros than poles.'
    self.CgainInDBInteract.value = mag2db(self.Kgain)
    if Gc.dt != None:
      assert all(np.abs(self.CPoles)<=1), 'Gc(z) should not have unstable poles.'
      assert all(np.abs(self.CZeros)<=1), 'Gc(z) should not have non minimum phase zeros.'
      omegaZ, omegaP = np.log(self.CZeros)/Gc.dt, np.log(self.CPoles)/Gc.dt
    else:
      assert all(np.real(self.CPoles)<=0), 'Gc(s) should not have unstable poles.'
      assert all(np.real(self.CZeros)<=0), 'Gc(s) should not have non minimum phase zeros.'
      omegaZ, omegaP = self.CZeros, self.CPoles

    for z in omegaZ:
      if np.abs(z-pzIntDiff)<1e-6: self.PolesAndZerosList.append(
              PoleOrZeroClass('zero','differentiator', self.Ts, self))
      elif np.imag(z)>0:
        wn, csi = np.abs(z), np.abs(np.real(z))/np.abs(z)
        self.PolesAndZerosList.append(
              PoleOrZeroClass('zero','complex', self.Ts, self, omega=wn, csi=csi))
      elif np.imag(z)==0: self.PolesAndZerosList.append(
          PoleOrZeroClass('zero','real', self.Ts, self, omega=-z))
    for p in omegaP:
      if np.abs(p-pzIntDiff)<1e-6:  self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','integrator', self.Ts, self))
      elif np.imag(p)>0: 
        wn, csi = np.abs(p), np.abs(np.real(p))/np.abs(p)
        self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','complex', self.Ts, self, omega=wn, csi=csi))
      elif np.imag(z)==0: self.PolesAndZerosList.append(
          PoleOrZeroClass('pole','real', self.Ts, self,omega=np.abs(p)))

  def insertPoleZero(self,b):
    if self.NewPZDropdown.value != ' ':
      PZtype_dict = {'integrator': ['pole','integrator'],
          'differentiator': ['zero','differentiator'],
          'real pole': ['pole','real'], 'real zero': ['zero','real'],
          'complex pole': ['pole','complex'],'complex zero': ['zero','complex']}
      PZtype, PZsubtype = PZtype_dict[self.NewPZDropdown.value]
      if PZtype == 'pole':
        if PZsubtype == 'complex': self.relatOrderC += 2
        else:                      self.relatOrderC += 1
      else:
        if PZsubtype == 'complex': self.relatOrderC -= 2
        else:                      self.relatOrderC -= 1
      self.PolesAndZerosList.append(PoleOrZeroClass(PZtype,PZsubtype,self.Ts,self))
      self.NewPZDropdown.value = ' '
      x = len(self.PolesAndZerosList)-1
      self.PolesAndZerosList[x].FrequencySetButton.on_click(self.updateTFAndBokeh)
    self.updateTFAndScreen(0)

  def printController(self,b):
    numstr, denstr = '', ''
    Kstr = f'{db2mag(self.CgainInDBInteract.value):.4f}'
    for pz in self.PolesAndZerosList:
      numstr = numstr + pz.printNumOrDen('num')
      denstr = denstr + pz.printNumOrDen('den')
    if len(numstr)<1: numstr = '1'
    if len(denstr)<1: denstr = '1'
    return str(numstr), str(denstr), Kstr
    
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
    _TTS_RLOC= [("real","@x"),("imag","@y"),('K','@K{0.00 a}')]
    _TTS_TRESP = [('signal', "$name"), ("t", "$x s"), ("value", "$y") ]
    self.figMag = BkFig(title="Bode Magnitude", plot_height=300, plot_width=400,
               toolbar_location="above", tooltips = _TTS_BD1, x_axis_type="log",
               x_axis_label='f (Hz)', y_axis_label='mag (dB)')
    self.figAng =  BkFig(title="Bode Angle", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_BD2, x_axis_type="log",
                x_axis_label='f (Hz)', y_axis_label='ang (°)')
    self.figAng.x_range  = self.figMag.x_range   #same axis
    self.figAng.yaxis.ticker=np.linspace(-720,720,17)
    self.figRLoc=  BkFig(title="Root Locus", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_RLOC,
                x_axis_label='real', y_axis_label='imag')
    #self.figRLoc.hover.line_policy = 'interp'
    self.figTResp = BkFig(title="Time Response", plot_height=300, plot_width=400,
                toolbar_location="above", tooltips = _TTS_TRESP,
                x_axis_label='time (s)', y_axis_label='y') 
    self.figTResp2= BkFig(title="Time Response", plot_height=300, plot_width=800, 
                toolbar_location="above", tooltips = _TTS_TRESP,
                x_axis_label='time (s)', y_axis_label='y')
 
    self.Bkgrid = bokeh.layouts.layout([[self.figMag, self.figRLoc],
                                        [self.figAng, self.figTResp]])


    if self.Ts in [None, 0.0]:   #continuous time
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
    self.Clbltxt.text = 'C(s) = ' if self.Ts in [None, 0.0] else 'C(z) = '
    self.Cgaintxt = Label(x=40, y=20, x_units='screen', y_units='screen', 
                         text='K',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Cnumtxt = Label(x=100, y=30, x_units='screen', y_units='screen', 
                         text='N',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.Cdentxt = Label(x=100, y=10, x_units='screen', y_units='screen', 
                         text='D',  render_mode='css',border_line_color=None,
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
    rlocuslinehv = self.figRLoc.line(x='x',y='y',line_alpha=0, 
                                  name='rlocus2', source = self.rlocussource)
    self.figRLoc.hover.renderers=[rlocuslinehv, rlocusGpoles, rlocusGzeros,
                                  rlocusCpoles, rlocusCzeros, rlocusMF]
    #self.figRLoc.hover.mode='mouse'   
    #self.figRLoc.hover.line_policy='next'
    #self.figRLoc.hover.point_policy='snap_to_data'
    self.Stabilitytxt = Label(x=10, y=200, x_units='screen', y_units='screen', 
                         text=' ',  render_mode='css',border_line_color=None,
                        background_fill_color='white',text_font_size = '11px')
    self.figRLoc.add_layout(self.Stabilitytxt)

    #Step response:
    self.figTResp.extra_y_ranges = {'u_range': bokeh.models.Range1d()}
    self.figTResp.add_layout(bokeh.models.LinearAxis(y_range_name="u_range",
                                                     axis_label='u'), 'right')
    self.figTResp.y_range = bokeh.models.Range1d(start = -0.1, end = 1.4)
    if self.Ts in [None, 0.0]:
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
    self.createRLocus()
    self.createBode()
    self.updateStepResponse()
    self.Appwidget.center = VBox([pzs.Widget for pzs in self.PolesAndZerosList])
    asdfs = display(self.Appwidget)
    bokeh.io.output_notebook()
    self.Bknb_handle = bokeh.io.show(self.Bkgrid, notebook_handle=True)
    self.updateBokeh()

  def updateGainAndBokeh(self,b):
    #Update gain:  updates Gc(s),  T(s), and Kgain
    Kgain_new = db2mag(self.CgainInDBInteract.value)
    dKgain = Kgain_new/self.Kgain
    self.CTransfFunc = self.CTransfFunc*dKgain
    self.OLTF = self.OLTF*dKgain
    self.Kgain, self.dKgaindB = Kgain_new,  np.round(mag2db(dKgain),decimals=1)
    #Update Bokeh:
    self.updateBokeh()

  def updateAndPrintC(self,b):
    self.updateTFAndBokeh(0)
    N,D,K = self.printController(0)
    print(f'Controller:  num = {self.numC}')
    print(f'             den  = {self.denC}')
    print(f'ZPK:  zeros = {self.CZeros}')
    print(f'      poles = {self.CPoles}')
    print(f'      gain = {self.Kgain}')

  def updateTFAndBokeh(self,b):
    self.updateTransferFunction()
    self.createBode()
    self.createRLocus()
    self.updateBokeh()

  def updateTFAndScreen(self,b):
    self.updateTransferFunction()
    #print(self.CTransfFunc)
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
    self.OLTF = minreal(self.GpTransfFunc*self.CTransfFunc,tol=1e-6,verbose=False)

  def createBode(self):
    '''Creates the plots for Bode Diagram '''
    magT,phiT,omega = bode(self.OLTF,omega_num=1000, plot=False)
    magG,phiG,_ = bode(self.GpTransfFunc,omega, plot=False)
    magdbG, magdbT = mag2db(magG), mag2db(magT)
    phiGHz, phiTHz = phiG*180/pi, phiT*180/pi
    self.bodesource.data={'omega':omega, 'freqHz':(omega/(2*np.pi)),
                         'magdBG':magdbG, 'magG':magG, 'angG':phiGHz,
                         'magdBT':magdbT, 'magT':magT, 'angT':phiTHz}
    self.updatePMGM()
    def d2c_clampAtNyquistFreq(PZdiscr):
          PZdiscr1 = list(filter(lambda x: np.real(x)>=0, PZdiscr))
          omegaVec = np.abs(np.log(PZdiscr1))/self.Ts
          omega_nyqu = 2*np.pi*self.fNyquistHz
          for q in range(len(omegaVec)):
            if omegaVec[q]>omega_nyqu: omegaVec[q] = omega_nyqu
          return omegaVec
    func1 = np.abs if self.Ts in [None, 0.0] else d2c_clampAtNyquistFreq
    dict1 = {'Gpp': [self.GpPoles,self.gpbodesource],
             'Gpz': [self.GpZeros,self.gzbodesource],
             'CP' : [self.CPoles,self.cpbodesource],
             'CZ' : [self.CZeros,self.czbodesource]}
    for key1 in ['Gpp','Gpz','CP','CZ']:
      pORz = list(filter(lambda x: x>1e-5, func1(dict1[key1][0])))
      magdB, phideg, fHz = [], [], []
      if pORz:
        mag,phi,omega = bode(self.OLTF, pORz, plot=False)
        magdB, phideg, fHz = mag2db(mag), phi*180/pi, (omega/(2*pi))
        for q in range(len(phideg)):
          if phideg[q] > 90: phideg[q] = phideg[q]-360;
      dict1[key1][1].data={'fHz':list(fHz),'magdB':list(magdB),'angdeg':list(phideg)}

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
    if str(self.GainMargin) == 'inf':  self.GMtxt.text = 'GM: inf'
    else: self.GMtxt.text = f'GM:{self.GainMargin:.1f} dB'
    if str(self.PhaseMargin) == 'inf': self.PMtxt.text = 'PM: inf'
    else: self.PMtxt.text = f'PM: {self.PhaseMargin:.1f}°'

  def createRLocus(self):
    CgaindB = self.CgainInDBInteract.value
    Cgain = db2mag(CgaindB)
    self.kvectLen= int((self.maxCGainIndB-self.minCGainIndB)/self.CGainStepIndB+1)
    kvectdB = np.linspace(self.minCGainIndB, self.maxCGainIndB, self.kvectLen)
    self.kvect = db2mag(kvectdB)
    Kgp, Kgz = np.zeros(len(self.GpPoles)), np.Inf*np.ones(len(self.GpZeros))
    Kcp, Kcz = np.zeros(len(self.CPoles)), np.Inf*np.ones(len(self.CZeros))
    self.rootsVect,_ = rlocus(self.OLTF/Cgain,plot=False,kvect=self.kvect)
    re, im, rootsVect = np.real, np.imag, self.rootsVect
    Krlocus,  cols = self.kvect,  self.rootsVect.shape[1]-1
    for x in range(cols):  Krlocus = np.column_stack((Krlocus,self.kvect))
    self.rlocussource.data = {'x':re(rootsVect),'y':im(rootsVect),'K':Krlocus}
    self.gprlocussource.data = {'x':re(self.GpPoles),'y':im(self.GpPoles),'K':Kgp}
    self.gzrlocussource.data = {'x':re(self.GpZeros),'y':im(self.GpZeros),'K':Kgz}
    self.cprlocussource.data = {'x':re(self.CPoles),'y':im(self.CPoles),'K':Kcp}
    self.czrlocussource.data = {'x':re(self.CZeros),'y':im(self.CZeros),'K':Kcz}
    self.updateRLocusData()

    if rootsVect.size>0: 
      xrangemin, xrangemax = np.min(re(rootsVect)), np.max(re(rootsVect))
      if np.abs(xrangemax-xrangemin)<2:
        self.figRLoc.x_range.update(start=xrangemin-1, end=xrangemax+1)
      yrangemin, yrangemax = np.min(im(rootsVect)), np.max(im(rootsVect))
      if np.abs(yrangemax-yrangemin)<2:
        self.figRLoc.y_range.update(start=yrangemin-1, end=yrangemax+1) 
    

  def updateRLocusData(self):
    Cgain_real = db2mag(self.CgainInDBInteract.value)
    Kindex = int(self.kvectLen*(self.CgainInDBInteract.value-self.minCGainIndB)
                                       /(self.maxCGainIndB-self.minCGainIndB))-1
    x,y = np.real(self.rootsVect[Kindex]), np.imag(self.rootsVect[Kindex])
    K = self.kvect[Kindex]*np.ones(len(list(x)))
    self.krlocussource.data = {'x': x , 'y': y, 'K':K} 
    comp = (x>0) if self.Ts in [None, 0.0] else  ((x*x+y*y)>1)
    if any(comp): self.Stabilitytxt.text = 'Unstable Loop'
    else:         self.Stabilitytxt.text = 'Stable Loop'

  def updateStepResponse(self):
    Gmf = minreal(feedback(self.OLTF, 1), tol=1e-6, verbose=False)
    Gru = feedback(self.CTransfFunc, self.GpTransfFunc)
    p_dom = np.abs(np.real(Gmf.pole()))
    wp_dom = p_dom if self.Ts in [None, 0.0] else -np.log(p_dom)/self.Ts
    tau5_Gmf = np.abs(5/np.min(wp_dom)) #5 constantes de tempo
    if self.Ts in [None, 0.0]:
      tvec = linspace(0,tau5_Gmf, 200)
    else:
      nmax = np.round(tau5_Gmf/self.Ts)
      tvec = linspace(0,nmax*self.Ts, int(nmax+1))
    ymf,tvec = step(Gmf, T=tvec)
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
    N,D,K = self.printController(0)
    self.Cnumtxt.text, self.Cdentxt.text, self.Cgaintxt.text = 'N','D','K'
    self.Cnumtxt.text, self.Cdentxt.text, self.Cgaintxt.text = N,D,K
