
class PoleOrZeroClass():
  csi0 = 0.707
  relatOrderC = 0
  from numpy import sqrt

  dict_TXT0func_TFc = {'': (lambda w,csi: ''),
                      'integrator': (lambda w,csi: 's'),
                      'differentiator': (lambda w,csi: 's'),
                      'real pole': (lambda w,csi: f'(s/{w:.4f} + 1)'),
                      'real zero': (lambda w,csi: f'(s/{w:.4f} + 1)'),
                      'complex pole': (lambda w,csi: f'(s²/{(w**2):.4f}{"" if csi==0 else f"+ s {2*csi:.4f}/{w:.4f}"} + 1)')  ,
                      'complex zero': (lambda w,csi: f'(s²/{(w**2):.4f}{"" if csi==0 else f"+ s {2*csi:.4f}/{w:.4f}"} + 1)')}
  dict_NUMfunc_TFc = {'': (lambda w,csi: [1]),
                      'integrator': (lambda w,csi: [0,1]),
                      'differentiator': (lambda w,csi: [1,0]),
                      'real pole': (lambda w,csi: [0,1]),
                      'real zero': (lambda w,csi: [1/w,1]),
                      'complex pole': (lambda w,csi: [0,0,1]),
                      'complex zero': (lambda w,csi: [1/w**2,2*csi/w,1])}
  dict_DENfunc_TFc = {'': (lambda w,csi: [1]),
                      'integrator': (lambda w,csi: [1,0]),
                      'differentiator':(lambda w,csi: [0,1]),
                     'real pole':  (lambda w,csi: [1/w,1]),
                      'real zero': (lambda w,csi: [0,1]),
                      'complex pole': (lambda w,csi: [1/w**2,2*csi/w,1]),
                      'complex zero': (lambda w,csi: [0,0,1])}
  dict_NUMfunc_TFd = {'': (lambda pz,a1,a2,Ts:[1]),
                      'integrator':(lambda pz,a1,a2,Ts: [0,1]),
                      'differentiator':(lambda pz,a1,a2,Ts: [1/Ts,-1/Ts]),
                      'real pole':  (lambda pz,a1,a2,Ts: [0,1]),
                      'real zero': (lambda pz,a1,a2,Ts: [1/(1-pz), -pz/(1-pz)]),
                      'complex pole': (lambda pz,a1,a2,Ts: [0,0,1]),
                      'complex zero': (lambda pz,a1,a2,Ts: [1/(1+a1+a2),a1/(1+a1+a2),a2/(1+a1+a2)]),
                      'unit delay': (lambda pz,a1,a2,Ts: [0,1]),
                      'unit advance': (lambda pz,a1,a2,Ts: [1,0]) }
  dict_DENfunc_TFd = {'': (lambda pz,a1,a2,Ts:[1]),
                      'integrator': (lambda pz,a1,a2,Ts: [1/Ts,-1/Ts]),
                      'differentiator': (lambda pz,a1,a2,Ts: [0,1]),
                      'real pole': (lambda pz,a1,a2,Ts: [1/(1-pz), -pz/(1-pz)]),
                      'real zero': (lambda pz,a1,a2,Ts: [0,1]),
                      'complex pole': (lambda pz,a1,a2,Ts: [1/(1+a1+a2),a1/(1+a1+a2),a2/(1+a1+a2)]),
                      'complex zero': (lambda pz,a1,a2,Ts: [0,0,1]),
                      'unit delay': (lambda pz,a1,a2,Ts: [1,0]),
                      'unit advance': (lambda pz,a1,a2,Ts: [0,1]) }
  dict_TXT0func_TFd = {'': (lambda pz,a1,a2,Ts: ''),
                      'integrator': (lambda pz,a1,a2,Ts: f'(z {1/Ts:.4e} {-1/Ts:.4e})'),
                      'differentiator': (lambda pz,a1,a2,Ts: f'(z {1/Ts:.4e} {-1/Ts:.4e})'),
                      'real pole': (lambda pz,a1,a2,Ts: f'(z {1/(1-pz):.4e} {-pz/(1-pz):.4e})'),
                      'real zero': (lambda pz,a1,a2,Ts: f'(z {1/(1-pz):.4e} {-pz/(1-pz):.4e})'),
                      'complex pole': (lambda pz,a1,a2,Ts: f'(z² {1/(1+a1+a2):.4e} - z {-a1/(1+a1+a2):.4e} + {a2/(1+a1+a2):.4e}'),
                      'complex zero': (lambda pz,a1,a2,Ts: f'(z² {1/(1+a1+a2):.4e} - z {-a1/(1+a1+a2):.4e} + {a2/(1+a1+a2):.4e}'),
                      'unit delay': (lambda pz,a1,a2,Ts: 'z'),
                      'unit advance': (lambda pz,a1,a2,Ts: 'z')}
  dict_Poles={'': (lambda den,Ts: []),
             'integrator': (lambda den,Ts: [(0 if Ts in [0,None] else 1)]),
             'differentiator':(lambda den,Ts: []),
             'real pole':(lambda den,Ts: [-den[1]/den[0]]),
              'real zero':(lambda den,Ts: []),
             'complex pole': (lambda den,Ts: [(-den[1]+1j*(4*den[0]*den[2]-den[1]**2)**0.5)/(2*den[0]) ,
                                              (-den[1]-1j*(4*den[0]*den[2]-den[1]**2)**0.5)/(2*den[0])]),
             'complex zero': (lambda den,Ts: []),
              'unit delay': (lambda den,Ts: ([] if Ts in [0,None] else [0])  ),
              'unit advance': (lambda den,Ts: [] )}
  dict_Zeros={'': (lambda num,Ts: []),
             'integrator': (lambda num,Ts: []),
              'differentiator':(lambda num,Ts: [(0 if Ts in [0,None] else 1)]),
             'real zero':(lambda num,Ts: [-num[1]/num[0]]),
              'real pole':(lambda num,Ts: []),
             'complex zero': (lambda num,Ts: [(-num[1]+1j*(4*num[0]*num[2]-num[1]**2)**0.5)/(2*num[0]) ,
                                              (-num[1]-1j*(4*num[0]*num[2]-num[1]**2)**0.5)/(2*num[0])]),
             'complex pole': (lambda num,Ts: []),
              'unit advance': (lambda num,Ts: ([] if Ts in [0,None] else [0])  ),
              'unit delay': (lambda num,Ts: [] ) }
  POLEtypes = ['integrator', 'real pole', 'complex pole', 'unit delay']
  ZEROtypes = ['differentiator', 'real zero', 'complex zero', 'unit advance']
  d_relatOrd = {'': 0, 'integrator': 1, 'real pole': 1, 'complex pole': 2, 'unit delay': 1,
            'differentiator': -1, 'real zero': -1, 'complex zero': -2, 'unit advance': -1}

  def __init__(self, Ts):
    from ipywidgets import Dropdown, Checkbox, BoundedFloatText, Button, Layout, Text, Label

    LayS = Layout(display='flex', align_items='stretch', width='50px')
    LayM = Layout(display='flex', align_items='stretch', width='110px')
    LayL = Layout(display='flex', align_items='stretch', width='300px')

    self.Ts = Ts
    self.relatOrderC = 0
    self.oldPZType = ''
    self.freq0_Hz = 100 if self.Ts in [0, None] else 0.1/self.Ts
    self.fNy_Hz = 1e6 if self.Ts in [0, None] else 0.5/self.Ts
    self.ENwgt = Checkbox(value=False, description='', disabled=False, indent=False, layout = LayS)
    self.ENwgt.observe(self.enable_disable_PZ, names='value')
    self.TYwgt = Dropdown( options=['','integrator', 'real pole', 'complex pole',
                                   'differentiator', 'real zero', 'complex zero'],
                           value='', description='',  disabled=True, layout = LayM)
    self.TYwgt.observe(self.change_PZtype, names='value')
    self.Fwgt = BoundedFloatText( value=self.freq0_Hz,  min=1e-3, max = self.fNy_Hz,
                                 continuous_update=True, step=1e-2,
                                 description='', disabled=True, layout = LayM)
    self.Fwgt.observe(self.change_freqOrCsi, names='value')
    self.CSIwgt = BoundedFloatText( value=self.csi0,  min=0, max=0.999, continuous_update=True,
                                step=1e-3,  description='', disabled=True,  layout = LayM)
    self.CSIwgt.observe(self.change_freqOrCsi, names='value')
    self.SETwgt = Button(description='Set',  disabled=True,
                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                        tooltip='Set', icon='check',
                        layout = Layout(display='flex', align_items='stretch', width='65px'))
    self.SETwgt.on_click(self.set_button_on_click)
    self.SETwgt.observe(self.set_button_changes_status, names='button_style')

    self.TXTwgt = Label(description='', disabled = True, layout = LayL)
    from numpy import array
    self.PZnum, self.PZden = array([1]), array([1])
    self.Zeros, self.Poles = array([]), array([])

  def enable_disable_PZ(self, dict_observe):
    disable = dict_observe['old']  #True to disable. False to enable
    self.SETwgt.button_style = ''  #first resets "set" button -> runs "set_button_changes_status"
    self.TYwgt.value = ''          #resets type. If changed -> runs  "change_PZtype"
    self.TYwgt.disabled = disable

  def change_PZtype(self, dict_observe):
    from numpy import array
    newPZType, oldPZType = dict_observe['new'], dict_observe['old']
    self.PZnum, self.PZden = array([1]), array([1])
    self.oldPZType = oldPZType
    self.SETwgt.button_style = ''  #first resets "set" button -> runs "set_button_changes_status"
    self.oldPZType = newPZType   #updates after using method "set_button_changes_status"
    self.SETwgt.disabled = True if newPZType == '' else False
    self.Fwgt.disabled = True if newPZType in ['','integrator','differentiator'] else False
    self.CSIwgt.disabled = False if newPZType in ['complex pole','complex zero'] else True
    self.TXTwgt.disabled = True if newPZType == '' else False
    self.Fwgt.value = self.freq0_Hz
    self.CSIwgt.value = self.csi0
    self.show_update_tf_text()

  def change_freqOrCsi(self, dict_observe):
    self.SETwgt.button_style = ''
    from numpy import array
    self.PZnum, self.PZden = array([1]), array([1])
    self.show_update_tf_text()

  def set_button_on_click(self, value):
    if self.SETwgt.button_style == '' and (    #roda somente quando deve mudar de estado
        (self.TYwgt.value in ['integrator', 'real pole', 'complex pole', 'unit delay'])
         or ( self.TYwgt.value in ['differentiator', 'real zero', 'unit advance']  and PoleOrZeroClass.relatOrderC>=1 )
         or ( self.TYwgt.value == 'complex zero'  and PoleOrZeroClass.relatOrderC>=2 ) ):
      #If everything is a success:
      self.SETwgt.button_style = 'success'  #->after this, runs "set_button_changes_status"
    else:
      print('Gc should not have more zeros than poles. First set the poles.')

  def set_button_changes_status(self,dict_observe):
    '''Changes the relative order or the controller '''
    oldStatus, newStatus = dict_observe['old'], dict_observe['new']
    if newStatus == 'success':
      PoleOrZeroClass.relatOrderC +=  self.d_relatOrd[self.oldPZType]  #increases relat orde
    else:  #if new status is ''
      PoleOrZeroClass.relatOrderC -=  self.d_relatOrd[self.oldPZType]  #removes relat order

  def show_update_tf_text(self):
    '''Runs always when anything changes'''
    from numpy import pi, exp, cos, sqrt, array
    w, csi = 2*pi*self.Fwgt.value,  self.CSIwgt.value
    if self.Ts in [0, None]:   #continuous-time system
      self.TXTwgt.value = ( ('1/' if self.TYwgt.value in self.POLEtypes else '')
                          + (self.dict_TXT0func_TFc[self.TYwgt.value])(w,csi)   )
      self.PZnum = array((self.dict_NUMfunc_TFc[self.TYwgt.value])(w,csi))
      self.PZden = array((self.dict_DENfunc_TFc[self.TYwgt.value])(w,csi))
    else:   #discrete-time systems
      a1, a2, pz = -2*exp(-csi*w*self.Ts)*cos(w*sqrt(1-csi**2)*self.Ts), exp(-2*csi*w*self.Ts), exp(-w*self.Ts)
      self.TXTwgt.value = ( ('1/' if self.TYwgt.value in self.POLEtypes else '')
                          + (self.dict_TXT0func_TFd[self.TYwgt.value])(pz,a1,a2,self.Ts)   )
      self.PZnum =  array((self.dict_NUMfunc_TFd[self.TYwgt.value])(pz,a1,a2,self.Ts))
      self.PZden =  array((self.dict_DENfunc_TFd[self.TYwgt.value])(pz,a1,a2,self.Ts))
    self.Poles = array( (self.dict_Poles[self.TYwgt.value])(self.PZden, self.Ts))
    self.Zeros = array( (self.dict_Zeros[self.TYwgt.value])(self.PZnum, self.Ts))
    #self.TXTwgt.value = str(self.Zeros) + ' / ' + str(self.Poles)

  def printNumOrDen(self, num_den_key):
    if self.SETwgt.button_style == '': return ''
    if num_den_key=='num' and self.TYwgt.value in self.POLEtypes: return ''
    if num_den_key=='den' and self.TYwgt.value in self.ZEROtypes: return ''
    from numpy import pi, exp, cos, sqrt, array
    w, csi = 2*pi*self.Fwgt.value,  self.CSIwgt.value
    if self.Ts in [0,None]:
      return (self.dict_TXT0func_TFc[self.TYwgt.value])(w,csi)
    else:
      a1 = -2*exp(-csi*w*self.Ts)*cos(w*sqrt(1-csi**2)*self.Ts)
      a2 = exp(-2*csi*w*self.Ts)
      pz = exp(-w*self.Ts)
      return (self.dict_TXT0func_TFd[self.TYwgt.value])(pz,a1,a2,self.Ts)


from control import TransferFunction

class ControllerSISOApp(TransferFunction):
  def __init__(self, num, den, dt = 0, name = 'C', inputs=['u1'], outputs=['y1'], start_configWgt = False, quantity_PZs = 4):
    from numpy import arange, real, log, round, array
    super().__init__(num,den,dt)
    self.input_labels = inputs
    self.output_labels = outputs
    self.name = name
    self.Poles = self.poles()
    self.Zeros = self.zeros()
    self._update_num_norm_den_norm()
    self._update_Kdcgain()
    self.set_by_zpk_minreal(self.Zeros,self.Poles, self.Kdcgain)
    #print(f'1) SZeros = {self.SZeros} \n SPoles = {self.SPoles}')

    if start_configWgt:  #initialization
      self.PZwidgets = []
      #Create ipywidgets layout and events:
      from ipywidgets import VBox, Label, HBox, FloatSlider, Button, HTML, Layout
      from control import db2mag
      ENBox, TYBox, FBox = [Label('Enable')], [Label('Type')], [Label('Frequency (Hz)')]
      CSIBox = [Label('ζ (damping ratio)')]
      self.SETBox =  [Button(description='Set All',  disabled=False, button_style='', tooltip='Set', icon='check',
                 layout = Layout(display='flex', align_items='stretch', width='65px')) ]
      TXTBox = [Label('Transfer Function')]

      for q in range(quantity_PZs):
          PoleOrZeroClass.relatOrderC = 0
          self.PZwidgets.append(PoleOrZeroClass(self.dt))
          ENBox.append(self.PZwidgets[q].ENwgt)
          TYBox.append(self.PZwidgets[q].TYwgt)
          FBox.append(self.PZwidgets[q].Fwgt)
          CSIBox.append(self.PZwidgets[q].CSIwgt)
          self.SETBox.append(self.PZwidgets[q].SETwgt)
          TXTBox.append(self.PZwidgets[q].TXTwgt)

      minCGainIndB, maxCGainIndB, CGainStepIndB = -80, 80, 0.5
      self.kvectdB = list(arange(minCGainIndB,maxCGainIndB, CGainStepIndB ))
      self.kvect = list(db2mag( array(self.kvectdB)))
      self.dKdcgain = 1
      self.CgainInDBInteract = FloatSlider(value=0, min=minCGainIndB,
                                       max=maxCGainIndB, step=CGainStepIndB,
                                       layout=Layout(width='410px',display='flex', align_items='stretch'),
                                       description = '',
                                       #description = '|G<sub>c</sub>| dB:',
                                       continuous_update=True)
      self.ControllerName = (f"G<sub>c</sub>({'s' if self.dt in {0,None} else 'z'}) = 1")
      self.ControllerText = HTML(value = self.ControllerName,
                                 layout=Layout(width='450px',display='flex', align_items='stretch'))
      self.Appwidget = VBox([HBox([VBox(ENBox),VBox(TYBox),VBox(FBox),VBox(CSIBox),VBox(self.SETBox),VBox(TXTBox)]),
                           HBox([HTML(r'|G<sub>c</sub>| dB:',layout=Layout(width='50px',display='flex', align_items='flex-start')),
                                 self.CgainInDBInteract, self.ControllerText])   ]  )

      #Put given controller to Widget:
      self._init_PZwidgets_by_given_controller()
      self.print_to_PZwidgets()
      if len(self.SPoles)>=1: self.SETBox[0].button_style = 'success'

      #Define events:
      self.CgainInDBInteract.observe(self.set_Kdcgain_by_PZwidgets,'value')
      for q in range(len(self.PZwidgets)):
        self.PZwidgets[q].SETwgt.observe(self.set_PZs_from_PZwidgets, names='button_style')
        self.PZwidgets[q].ENwgt.observe(self.enable_change_clear_set_all, names='value')
      self.SETBox[0].on_click(self.set_all_on_click)

  def print_latex(self,latex_raw_equation):
    from IPython.display import Math, HTML
    display(HTML(r"<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/latest.js?config=default'></script>"))
    return Math(latex_raw_equation)

  def _init_PZwidgets_by_given_controller(self):
    '''Inits the PZwidgets by a given controller'''
    from numpy import abs, real, imag, pi
    from control import mag2db, db2mag
    assert (len(self.Poles)+len(self.Zeros)<=len(self.PZwidgets)), 'Increase variable "quantity_PZs".'
    for q in range(len(self.PZwidgets)):
        self.PZwidgets[q].ENwgt.value = False   #first disables everything
    p_integrator = 0 if self.dt in [None, 0.0] else 1.0
    q = 0
    for p in self.SPoles:
        self.PZwidgets[q].ENwgt.value = True
        if abs(p-p_integrator)<1e-6:
          self.PZwidgets[q].TYwgt.value = 'integrator'
        elif imag(p)>1e-6:
          self.PZwidgets[q].TYwgt.value = 'complex pole'
          self.PZwidgets[q].Fwgt.value =  abs(p)/(2*pi)
          self.PZwidgets[q].CSIwgt.value = abs(real(p))/abs(p)
        else:
          self.PZwidgets[q].TYwgt.value = 'real pole'
          self.PZwidgets[q].Fwgt.value = abs(p)/(2*pi)
        self.PZwidgets[q].SETwgt.button_style = 'success'
        q += 1
    for z in self.SZeros:
        self.PZwidgets[q].ENwgt.value = True
        if abs(z-p_integrator)<1e-6:
          self.PZwidgets[q].TYwgt.value = 'differentiator'
        elif imag(z)>1e-6:
          self.PZwidgets[q].TYwgt.value = 'complex zero'
          self.PZwidgets[q].Fwgt.value =  abs(z)/(2*pi)
          self.PZwidgets[q].CSIwgt.value = abs(real(z))/abs(z)
        else:
          self.PZwidgets[q].TYwgt.value = 'real zero'
          self.PZwidgets[q].Fwgt.value = abs(z)/(2*pi)
        self.PZwidgets[q].SETwgt.button_style = 'success'
        q += 1
    self._update_num_norm_den_norm()
    self._update_Kdcgain()
    self.CgainInDBInteract.value = mag2db(self.Kdcgain)
    #self.set_PZs_from_PZwidgets(0)

  def set_PZs_from_PZwidgets(self, button_style):
    '''Runs when the "Set" Buttons change
       Sets the controller by pulling data from the PZ widgets'''
    if button_style['new']=='success':
      #Here inserts all poles and zeros with 'success'
      notcompletePZs = [ (self.PZwidgets[q].ENwgt.value ^ (self.PZwidgets[q].SETwgt.button_style=='success')) for q in range(len(self.PZwidgets))]
      #if not any(notcompletePZs):  #if all complete
      self.SETBox[0].button_style = 'success' if not any(notcompletePZs) else '' #"Set All" Button = all success
    else:     #not success
      self.SETBox[0].button_style = ''
      if PoleOrZeroClass.relatOrderC<0:  #if problems when removing
        for q in range(len(self.PZwidgets)):
            self.PZwidgets[q].SETwgt.button_style = '' #unsets everything ->
    #Set the Poles and Zeros:
    from numpy import array, concatenate
    Poles, Zeros = array([]), array([])
    for q in range(len(self.PZwidgets)):
      if self.PZwidgets[q].SETwgt.button_style == 'success':
          Poles = concatenate((Poles, self.PZwidgets[q].Poles))
          Zeros = concatenate((Zeros, self.PZwidgets[q].Zeros))
    self.set_by_zpk_minreal(Zeros,Poles,self.Kdcgain)
    self.print_to_PZwidgets()

  def set_Kdcgain_by_PZwidgets(self, b):
    from control import db2mag, mag2db
    Kdcgain_old = self.Kdcgain
    Kdcgain_new = db2mag(self.CgainInDBInteract.value)
    self.dKdcgain = Kdcgain_new/Kdcgain_old
    self.Kdcgain = Kdcgain_new
    self.print_to_PZwidgets()

  def print_to_PZwidgets(self):
    numstr, denstr ='', ''
    for q in range(len(self.PZwidgets)):
      numstr = numstr + self.PZwidgets[q].printNumOrDen('num')
      denstr = denstr + self.PZwidgets[q].printNumOrDen('den')
    if len(numstr)<1: numstr = '1'
    if len(denstr)<1: denstr = '1'
    equation_string = r'{self.Kdcgain:.4f} \frac{ numstr }{denstr}'
    self.ControllerText.value = f"G<sub>c</sub>({'s' if self.dt in {0,None} else 'z'}) = {self.Kdcgain:.4f} <sup>{numstr}</sup>&frasl;<sub>{denstr}</sub>"

  def set_by_zpk_minreal(self, Zeros, Poles, Kdcgain):
    r""" Run after setting Poles and Zeros in the App
    Gc(s) = Gain*
    :math: `G_c(s) = K_{gain} \frac{\prod_i (s/\omega_{zi} + 1)}{\prod_i (s/\omega_{pi} + 1)}`
    """
    from numpy import array, round
    self.Kdcgain = Kdcgain
    Poles, Zeros = round(Poles,6), round(Zeros,6)
    common_values = set(Zeros) & set(Poles)  #find commonn poles and zeros
    Zeros, Poles = list(Zeros), list(Poles)
    for value in common_values:
      Zeros.remove(value)  #removes equal poles and zeros
      Poles.remove(value)
    self.Zeros, self.Poles = array(Zeros), array(Poles)
    self._update_num_norm_den_norm()
    self._update_relatOrder()
    self._update_SPoles_SZeros()
    super().__init__(self.Kdcgain*self.num_norm , self.den_norm, self.dt)

  def _update_relatOrder(self):
    '''After updating  self.Zeros and self.Poles'''
    self.relatOrderC = len(self.Poles)-len(self.Zeros)
    PoleOrZeroClass.relatOrderC = self.relatOrderC
    assert self.relatOrderC>=0, 'Gc should not have more zeros than poles.'

  def _update_SPoles_SZeros(self):
    '''After updating  self.Zeros and self.Poles
      SPoles and SZeros are in rad/s
    '''
    from numpy import seterr, log, round, real
    seterr(divide = 'ignore', invalid = 'ignore')
    self.SPoles = self.Poles if self.dt in [0,None] else round(log(self.Poles)/self.dt,6)  #z = exp(s*Ts)
    self.SZeros = self.Zeros if self.dt in [0,None] else round(log(self.Zeros)/self.dt,6)
    seterr(divide = 'warn', invalid = 'warn')
    #assert all(real(self.SPoles)<=0), 'Gc(s) should not have unstable poles.'
    #assert all(real(self.SZeros)<=0), 'Gc(s) should not have non minimum phase zeros.'

  def _update_num_norm_den_norm(self):
    '''After updating  self.Zeros and self.Poles'''
    from numpy import abs, array, flipud, round, convolve, sum, real
    from numpy.polynomial.polynomial import polyfromroots
    p_integrator = 0 if self.dt in [None, 0.0] else 1.0
    Zeros_filt = list(filter(lambda x: abs(x-p_integrator)>=1e-6, self.Zeros ))
    quantity_differentiators = len(self.Zeros)- len(Zeros_filt)
    Poles_filt = list(filter(lambda x: abs(x-p_integrator)>=1e-6, self.Poles ))
    quantity_integrators = len(self.Poles) - len(Poles_filt)
    num_filt = flipud(polyfromroots(Zeros_filt))
    den_filt = flipud(polyfromroots(Poles_filt))
    if self.dt in [0, None]:    #continuous-time
      num_filt /= num_filt[-1]
      den_filt /= den_filt[-1]
      num_diff = array([1] + quantity_differentiators*[0])
      den_int = array([1] + quantity_integrators*[0])
    else:                  #discrete-time
      num_filt /= sum(num_filt)
      den_filt /= sum(den_filt)
      num_diff, den_int = array([1]), array([1])
      for q in range(quantity_differentiators):  num_diff = convolve(num_diff,[1,-1])
      for q in range(quantity_integrators):  den_int = convolve(den_int,[1,-1])
    self.num_norm = real(round(convolve(num_filt, num_diff),12))
    self.den_norm = real(round(convolve(den_filt, den_int),12))

  def _update_Kdcgain(self):
    '''Works only after running "update_num_norm_den_norm" '''
    from scipy.signal import tf2zpk
    from numpy import real, round
    _,_,k1 = tf2zpk(self.num_norm, self.den_norm)
    _,_,k2 = tf2zpk(self.num[0][0], self.den[0][0])
    self.Kdcgain = round(real(k2/k1),12)

  def enable_change_clear_set_all(self,b):
    self.SETBox[0].button_style = ''
    if self.SETBox[0].button_style == '':   #if all enabled others are ready:
      notcompletePZs = [ (self.PZwidgets[q].ENwgt.value ^ (self.PZwidgets[q].SETwgt.button_style=='success')) for q in range(len(self.PZwidgets))]
      if not any(notcompletePZs):  #if all complete
        self.SETBox[0].button_style = 'success'  #"Set All" Button = success

  def set_all_on_click(self, value):
    for q in range(len(self.PZwidgets)):
      if self.PZwidgets[q].TYwgt.value in PoleOrZeroClass.POLEtypes:  #poles
        self.PZwidgets[q].set_button_on_click(0)
    for q in range(len(self.PZwidgets)):  #zeros
      if self.PZwidgets[q].TYwgt.value in PoleOrZeroClass.ZEROtypes:   #zeros
        self.PZwidgets[q].set_button_on_click(0)


import matplotlib.pyplot as plt


class DisturbancesWidget():
  freq0_Hz = 1
  tmax_s = 10
  dt_s   = 1e-3

  dic_configs = {'': {'f': [True, 0],  't': [True,''],       'a':[True,''],        'dadt': [True, 0]   },  #[disabled, value]
            'steps': {'f': [True, 0],  't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [False, 1e9]},  #[disabled, value]
             'sine': {'f': [False,60], 't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [False, 1e9]},  #[disabled, value]
           'square': {'f': [False,60], 't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [False, 1e9]},  #[disabled, value]
         'triangle': {'f': [False,60], 't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [False, 1e9]   },  #[disabled, value]
              'PWL': {'f': [True,0],   't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [True, 0]   },  #[disabled, value]
            'P-PWL': {'f': [True,0],   't': [False,'0, 1'],  'a':[False,'1, 0.9'], 'dadt': [True, 0]   },  #[disabled, value]
            'noise': {'f':[False,1000],'t': [False,'0'],     'a':[False,'1'],      'dadt': [False, 1e9]}}  #[disabled, value]
  dic_description = {'': '',
                'steps': '"Amplitude" is set at given "time" points.',
                 'sine': '"Amplitude" is set at given "time" points. "Amplitude" can be a complex number (a+bj) in order to change the angle.',
               'square': '"Amplitude" is set at given "time" points.',
             'triangle': '"Amplitude" is set at given "time" points.',
                  'PWL': 'Piecewise Linear Function: set "time" and "amplitude" points.',
                'P-PWL': 'Peridic Piecewise Linear Function: set "time" and "amplitude" points. Period = last "Time point".',
                'noise': '"Amplitude points" is the standard deviation.   "Freq (Hz)": BandWidth for 1st order LPF. '}
  def __init__(self, tmax_s = 10, dt_s = 1e-3, freq0_Hz = 1):
    from numpy import array, arange, zeros_like
    self.tmax_s, self.dt_s, self.freq0_Hz = tmax_s, dt_s, freq0_Hz
    self.timeVec_s = arange(0, self.tmax_s, self.dt_s)
    self.waveVec = zeros_like(self.timeVec_s)
    self.Time_points = array([0])
    self.Amplitude_points = array([0])

    from ipywidgets import Dropdown, Checkbox, BoundedFloatText, Button, Layout, Text, Label

    LayS = Layout(display='flex', align_items='stretch', width='50px')
    LayS1 =  Layout(display='flex', align_items='stretch', width='65px')
    LayM = Layout(display='flex', align_items='stretch', width='100px')
    LayM1 = Layout(display='flex', align_items='stretch', width='120px')
    LayL = Layout(display='flex', align_items='stretch', width='300px')
    self.ENwgt = Checkbox(value=False, description='', disabled=False, indent=False, layout = LayS)
    self.INwgt = Dropdown( options=['','r', 'du', 'dy', 'dm'],
                           value='', description='',  disabled=True, layout = LayS)
    self.TYwgt = Dropdown( options=list(DisturbancesWidget.dic_configs.keys()),  #_|‾ ∿  ⎍  ⌵⌵
                           value='', description='',  disabled=True, layout = LayM)
    self.Fwgt = BoundedFloatText( value=self.freq0_Hz,  min=0, max = 1e6,
                                 continuous_update=True, step=1e-2,
                                 description='', disabled=True, layout = LayS1)
    self.Tvecwgt = Text( value='',  continuous_update=True, description='',
                         disabled=True,  layout = LayM1)
    self.Avecwgt = Text( value='',  continuous_update=True, description='',
                         disabled=True,  layout = LayM1)
    self.dAdtwgt = BoundedFloatText( value=999999999,  min=0, max = 999999999,
                                 continuous_update=True, step=1e-1,
                                 description='', disabled=True, layout = LayM)
    self.SETwgt = Button(description='Set',  disabled=True,
                        button_style='', # 'success', 'info', 'warning', 'danger' or ''
                        tooltip='Set', icon='check', layout = LayS1)
    self.TXTwgt = Text(description='', disabled = False, layout = LayL)

    #Events:
    self.ENwgt.observe(self.changes_Enable, names = 'value')
    self.INwgt.observe(self.changes_SignalInput, names = 'value')
    self.TYwgt.observe(self.changes_SignalType, names='value')
    self.Fwgt.observe(self.changes_FreqOrAmpOrTimeOrdAmpdt, names='value')
    self.Tvecwgt.observe(self.changes_FreqOrAmpOrTimeOrdAmpdt, names='value')
    self.Avecwgt.observe(self.changes_FreqOrAmpOrTimeOrdAmpdt, names='value')
    self.dAdtwgt.observe(self.changes_FreqOrAmpOrTimeOrdAmpdt, names='value')
    self.SETwgt.on_click(self.clicked_set_button)
    self.SETwgt.observe(self.changes_set_button, names='button_style')

  def changes_Enable(self,b):
    self.SETwgt.disabled = not b['new']
    self.INwgt.value = ''
    self.INwgt.disabled = not b['new']
    self.TYwgt.value = ''
    self.TYwgt.disabled = not b['new']

  def changes_SignalType(self,b):
    new_type = b['new']
    self.Fwgt.disabled =    self.dic_configs[new_type]['f'][0]
    self.Fwgt.value =       self.dic_configs[new_type]['f'][1]
    self.Tvecwgt.disabled = self.dic_configs[new_type]['t'][0]
    self.Tvecwgt.value =    self.dic_configs[new_type]['t'][1]
    self.Avecwgt.disabled = self.dic_configs[new_type]['a'][0]
    self.Avecwgt.value =    self.dic_configs[new_type]['a'][1]
    self.dAdtwgt.disabled = self.dic_configs[new_type]['dadt'][0]
    self.dAdtwgt.value =    self.dic_configs[new_type]['dadt'][1]
    self.TXTwgt.value =     self.dic_description[new_type]
    self.SETwgt.button_style = ''

  def changes_SignalInput(self,b):
    self.SETwgt.button_style = ''

  def changes_FreqOrAmpOrTimeOrdAmpdt(self,b):
    self.SETwgt.button_style = ''

  def clicked_set_button(self,b):
    from numpy import matrix, asarray, diff, all, insert, real
    if (self.SETwgt.button_style == '' and self.TYwgt.value != '' and self.INwgt.value != ''):
      self.Time_points = asarray(matrix(self.Tvecwgt.value))[0]
      self.Amplitude_points = asarray(matrix(self.Avecwgt.value))[0]
      if self.TYwgt.value != 'sine': self.Amplitude_points = real(self.Amplitude_points)
      if ((len(self.Time_points) == len(self.Amplitude_points)) and  #same quantity of points
           all(self.Time_points[:-1] < self.Time_points[1:]) and
            self.Time_points[0]>=0):
        self.SETwgt.button_style = 'success'
        if self.Time_points[0] != 0:
              self.Time_points = insert(self.Time_points,0,0)
              self.Amplitude_points = insert(self.Amplitude_points,0,0)
      else:
        print('"Amplitude" and "Time" must have the same quantity of points. "Time points" must be positive and in ascending order.')
        self.SETwgt.button_style = 'warning'

  def changes_set_button(self,b):
    new_style, old_style = b['new'], b['old']
    #print('Set button changed status')
    if new_style == 'success':
      self.create_waveVec(0)  #Creates Waveform Vector

  def create_waveVec(self,b):
    from numpy import array, zeros_like, piecewise, logical_and, any, arange
    from scipy.signal import sawtooth
    Amp_funcs, Condlist = [], []
    self.timeVec_s = arange(0, self.tmax_s, self.dt_s)
    if self.TYwgt.value in ['steps','sine','square','triangle', 'noise']:
        tf_dAdt = self.Time_points[0]
        for q in range(len(self.Amplitude_points)-1):
          #1) Includes constant part:
          condition = logical_and( self.timeVec_s >= tf_dAdt , self.timeVec_s < self.Time_points[q+1])
          if any(condition):
              Condlist.append( condition )
              Amp_funcs.append( (lambda t, c = self.Amplitude_points[q]: c) )
          #2) Includes dAdt parts:
          Ampl_init = Amp_funcs[-1](self.Time_points[q+1])  #if appended: const amplitude. Else: lambda func
          conditionVec, funcdAdt, tf_dAdt = self._return_dAdt_condition_func_tfinal(self.Time_points[q+1], Ampl_init, self.Amplitude_points[q+1])
          if any(conditionVec):
              Condlist.append(conditionVec)
              Amp_funcs.append(funcdAdt)
        #Last point:
        Condlist.append( self.timeVec_s >= tf_dAdt )
        Amp_funcs.append( (lambda t, c=self.Amplitude_points[-1]: c) )

    elif  self.TYwgt.value in ['PWL','P-PWL']:
        if (self.TYwgt.value == 'P-PWL') and (self.Time_points[-1] < self.tmax_s):
            self.timeVec_s = arange(0, self.Time_points[-1], self.dt_s)  #reduces self.timeVec_s to fit into the repeating period
        for q in range(len(self.Amplitude_points)-1):
          conditionVec, funcPWL = self._return_PWL_condition_func(self.Time_points[q], self.Time_points[q+1], self.Amplitude_points[q], self.Amplitude_points[q+1])
          Condlist.append(conditionVec)
          Amp_funcs.append(funcPWL)
        Amp_funcs.append(lambda t: self.Amplitude_points[-1])
    tvec = self.timeVec_s if self.TYwgt.value != 'sine' else self.timeVec_s.astype('complex128')
    self.waveVec = piecewise(tvec, Condlist, Amp_funcs)

    if self.TYwgt.value == 'sine':
        from numpy import real, imag, pi, sin, cos
        a, b, w = real(self.waveVec), imag(self.waveVec),  2*pi*self.Fwgt.value
        self.waveVec = a*sin(w*self.timeVec_s) - b*cos(w*self.timeVec_s)
    elif self.TYwgt.value in ['square', 'triangle']:
        from scipy.signal import sawtooth, square
        from numpy import pi
        func = square if self.TYwgt.value == 'square' else sawtooth
        self.waveVec = self.waveVec*func(2*pi*self.Fwgt.value*self.timeVec_s, 0.5)
    elif self.TYwgt.value == 'P-PWL':  #post-processing periodic signal
        from numpy import tile
        repeating_times = int(self.tmax_s/self.Time_points[-1])+1
        self.timeVec_s = arange(0, self.tmax_s, self.dt_s)
        self.waveVec = (tile(self.waveVec, repeating_times))[:len(self.timeVec_s)]
    elif self.TYwgt.value == 'noise':
        from numpy import random, pi, std
        from scipy.signal import lfilter
        noise_amp1 = random.normal(size = len(self.timeVec_s))
        if (self.Fwgt.value < 0.5/self.dt_s):  #Nyquist   #filter
          wTs = 2*pi*self.Fwgt.value*self.dt_s
          noise_amp1 = lfilter([1, 1],[(1+2/wTs), (1-2/wTs)], noise_amp1)
          noise_amp1 = noise_amp1/std(noise_amp1)
        #change amplitude over time:
        self.waveVec = self.waveVec*noise_amp1

  def _return_dAdt_condition_func_tfinal(self, time_init_s, Ampl_init, Ampl_final):
    from numpy import logical_and, abs
    if self.dAdtwgt.value == self.dAdtwgt.max:
      time_final_s = time_init_s
      conditionVec = [False]  #returns False
      func = lambda t, Ai=Ampl_init:  Ai
    else:
      time_final_s = time_init_s + abs(Ampl_final-Ampl_init)/self.dAdtwgt.value
      conditionVec = logical_and( self.timeVec_s >= time_init_s, self.timeVec_s < time_final_s)
      signal_direction = (Ampl_final-Ampl_init)/abs(Ampl_final-Ampl_init)
      dAdt = self.dAdtwgt.value*signal_direction
      func = lambda t, Ai=Ampl_init, ti=time_init_s, dadt=dAdt: Ai+(t-ti)*dadt
    return conditionVec,  func, time_final_s

  def _return_PWL_condition_func(self, time_init_s, time_final_s, Ampl_init, Ampl_final):
    from numpy import logical_and
    alpha = (Ampl_final-Ampl_init)/(time_final_s-time_init_s)
    conditionVec = logical_and( self.timeVec_s >= time_init_s, self.timeVec_s < time_final_s)
    return conditionVec,  lambda t, Ai=Ampl_init, ti=time_init_s, dAdt=alpha:   Ai+(t-ti)*dAdt


class ControlAnalysisWidget():
  def __init__(self, tmax_s = 10, dt_s = 1e-5, freq0_Hz = 1, discrete=False, quantity_disturbances = 4):
    from ipywidgets import BoundedFloatText, HBox, VBox, Layout, Label, Dropdown, Checkbox, Button, Text
    from numpy import array

    self.waveVec_dict = {'t_s': array([]), 'r': array([]), 'du': array([]),
                         'dy': array([]), 'dm': array([]), 'u': array([]), 'y':array([])}

    #Requirements:
    self.OShotIn = BoundedFloatText(value = 10, min=0, max=100, continuous_update=False,
                            layout = Layout(width='100px'))
    self.RTimeIn = BoundedFloatText(value = 0.1, min=0, continuous_update=False,
                             layout = Layout(width='100px'))
    self.STimeIn = BoundedFloatText(value = 0.1, min=0, continuous_update=False,
                             layout = Layout(width='100px'))
    self.RequirementsTab = HBox([VBox([Label('Max Overshot (%)'), self.OShotIn  ])  ,
                           VBox([Label('Max Rise Time (s)'), self.RTimeIn  ]) ,
                           VBox([Label('Max Settling Time (s)'), self.STimeIn])] )

    #Input signals
    self.DistWgts = []  #DisturbancesWidget
    self.ENbox, self.INbox = [Label('Enable')]         , [Label('Input')]
    self.TYbox, self.Fbox  = [Label('Signal')]         , [Label('Freq (Hz)')]
    self.Tbox,  self.Abox  = [Label('Time points (s)')], [Label('Amplitude points')]
    self.dAdtbox = [Label('Max Ampl/s')]
    self.SETAllButton = Button(description='Set All',  disabled=False, button_style='', tooltip='Set', icon='check',
                                layout=Layout(align_items='stretch', width='65px') )

    self.SETbox = [self.SETAllButton]
    self.TXTbox = [Label('Description')]
    for q in range(quantity_disturbances):
        self.DistWgts.append(DisturbancesWidget())
        self.ENbox.append(self.DistWgts[q].ENwgt)
        self.INbox.append(self.DistWgts[q].INwgt)
        self.TYbox.append(self.DistWgts[q].TYwgt)
        self.Fbox.append(self.DistWgts[q].Fwgt)
        self.Tbox.append(self.DistWgts[q].Tvecwgt)
        self.Abox.append(self.DistWgts[q].Avecwgt)
        self.dAdtbox.append(self.DistWgts[q].dAdtwgt)
        self.SETbox.append(self.DistWgts[q].SETwgt)
        self.TXTbox.append(self.DistWgts[q].TXTwgt)

    self.TotalTimeWgt = BoundedFloatText(description = 'Total Time (s)',value = tmax_s,
                      min=0, max=1e6, continuous_update=False, layout = Layout(width='180px'))
    self.StepTimeWgt = BoundedFloatText(description = 'Step Time (s)', value = dt_s, min=0, max=1, continuous_update=False,
                        layout = Layout(width='180px'), disable = discrete)
    self.PrintTimeWgt = BoundedFloatText(description = 'Print Time (s)', value = 0, min=0, max=10, continuous_update=False,
                        layout = Layout(width='180px'))
    self.SetTimeWgt =  Button(description='Set',  disabled=False, button_style='', tooltip='Set', icon='check',
                                layout=Layout(align_items='stretch', width='65px') )
    self.TimeconfigWgt = HBox([self.TotalTimeWgt, self.StepTimeWgt, self.PrintTimeWgt, self.SetTimeWgt])
    self.DistConfigApp = HBox([VBox(self.ENbox),VBox(self.INbox),VBox(self.TYbox),
                         VBox(self.Fbox),VBox(self.Tbox),VBox(self.Abox),
                         VBox(self.dAdtbox),VBox(self.SETbox),VBox(self.TXTbox)])
    self.DisturbSimulationTab = VBox([self.TimeconfigWgt, self.DistConfigApp])

    #Define events:
    for q in range(len(self.DistWgts)):
        self.DistWgts[q].SETwgt.observe(self.verify_complete_disturbs, names='button_style')
        self.DistWgts[q].ENwgt.observe(self.enable_change_clear_set_all, names='value')
    self.SETAllButton.on_click(self.set_all_on_click)
    self.SETAllButton.observe(self.set_disturbs_to_SisoApp, names = 'button_style')
    self.SetTimeWgt.on_click(self.set_time_configs_to_App)
    self.set_time_configs_to_App(0)
    self.TotalTimeWgt.observe(self.reset_set_time_button, names =  'value')
    self.StepTimeWgt.observe(self.reset_set_time_button,  names = 'value')
    #Events requirements:
    self.RTimeIn.observe(self.adjust_SettlingTime_requirements,  names = 'value')

  def enable_change_clear_set_all(self,b):
    self.SETAllButton.button_style = ''
    if self.SETAllButton.button_style == '':   #if all enabled others are ready:
      notcompletePZs = [ (self.DistWgts[q].ENwgt.value ^ (self.DistWgts[q].SETwgt.button_style=='success')) for q in range(len(self.DistWgts))]
      if not any(notcompletePZs):  #if all complete
        self.SETAllButton.button_style = 'success'  #"Set All" Button = success

  def set_all_on_click(self, value):
    self.SETAllButton.button_style = ''
    for q in range(len(self.DistWgts)):
      self.DistWgts[q].clicked_set_button(0)

  def verify_complete_disturbs(self, button_style):
    if button_style['new']=='success':
      notcompletes = [ (self.DistWgts[q].ENwgt.value ^ (self.DistWgts[q].SETwgt.button_style=='success')) for q in range(len(self.DistWgts))]
      self.SETAllButton.button_style = 'success' if not any(notcompletes) else '' #"Set All" Button = all success
    else:     #not success
      self.SETAllButton.button_style = ''

  def set_disturbs_to_SisoApp(self, button_style):
    '''When Set All Button is turned on '''
    if button_style['new']=='success':
        from numpy import array, concatenate, zeros_like, real
        tvec = real(self.DistWgts[0].timeVec_s)
        self.waveVec_dict  = {'t_s': tvec, 'r': zeros_like(tvec), 'du': zeros_like(tvec), 'dy': zeros_like(tvec),
                              'dm': zeros_like(tvec), 'u': zeros_like(tvec), 'y': zeros_like(tvec)}
        for q in range(len(self.DistWgts)):
            if self.DistWgts[q].SETwgt.button_style == 'success':
              key = self.DistWgts[q].INwgt.value
              self.waveVec_dict[key] = self.waveVec_dict[key] + self.DistWgts[q].waveVec

  def set_time_configs_to_App(self, b):
    from numpy import arange
    self.StepTimeWgt.max = min(self.StepTimeWgt.max, 0.01*self.TotalTimeWgt.value)
    self.PrintTimeWgt.max = min(self.PrintTimeWgt.max , self.TotalTimeWgt.value)
    DisturbancesWidget.tmax_s = self.TotalTimeWgt.value
    DisturbancesWidget.dt_s = self.StepTimeWgt.value
    for q in range(len(self.DistWgts)):
      self.DistWgts[q].tmax_s = self.TotalTimeWgt.value
      self.DistWgts[q].dt_s = self.StepTimeWgt.value
      self.DistWgts[q].SETwgt.button_style = ''
    self.set_all_on_click(0)
    self.SetTimeWgt.button_style = 'success'

  def reset_set_time_button(self,b):
      self.SetTimeWgt.button_style=''

  def adjust_SettlingTime_requirements(self,b):
    self.STimeIn.min = max(self.STimeIn.min, self.RTimeIn.value)


A = ControlAnalysisWidget()
#A.DisturbSimulationTab
#A.SETbox

class CtrSysInfo():
  def __init__(self, Gp, Gc, Gf):
    from ipywidgets import BoundedFloatText, HBox, VBox, Layout, Label, Button, Textarea, Text
    from scipy.signal import tf2zpk
    self.TypeLabel = Label('Continuous-time system' if Gp.dt in [0, None] else f'Discrete-time system: Ts = {Gp.dt} s.')
    self.GpLabel = Textarea(str(Gp),description='G<sub>p</sub> =',
                            disabled=True,layout=Layout(height="120px"))
    self.GfLabel = Textarea(str(Gf),description='G<sub>f</sub> =',
                            disabled=True,layout=Layout(height="120px"))
    self.GcLabel = Textarea(str(Gc),description='G<sub>c</sub> =',
                            disabled=True,layout=Layout(height="120px"))
    zeros,poles,kgain = tf2zpk(Gc.num[0][0] , Gc.den[0][0])
    s = 's' if Gp.dt in [0, None] else 'z'
    self.Widget = VBox([self.TypeLabel,
                        HBox([self.GpLabel,self.GfLabel,self.GcLabel]),
                        Label('Controller numerator and denominator:'),
                        HBox([Text(str(Gc.num[0][0]), description='G<sub>c</sub> num ='),
                             Text(str(Gc.den[0][0]), description='G<sub>c</sub> den =')]),
                        Label(f'Controller zpkdata:  Gc = k  Π({s}-zi) / Π({s}-pi)'),
                        HBox([Text(str(zeros), description='G<sub>c</sub> zeros ='),
                              Text(str(poles), description='G<sub>c</sub> poles ='),
                              Text(str(kgain), description='G<sub>c</sub> k gain =')])])


import numpy as np
import bokeh

class SISOApp:
  '''
  Class of SISO Design Python Application
      Control package:   import control.matlab as *
      GUI:  ipywidgets + Bokeh
  Initialization:
      SISOApp(Gp, Gc, Gf)
             Gp: plant transfer function: continuous-time or discrete-time;
                  The controlleri is composed of two parts:
                      - dynamic part with gain equal to 1
                      - static part with gain equal to k
             Gc (optional): controller transfer function
             Gf (optional - NOT IMPLEMENTED YET): sensor transfer function

        Control structure of a SISO system:
            r: reference
            u: controller output
            y: process output
            du, dy, dm:  input, output and measurement disturbances

                          du↓+             dy↓+
     r ──→+◯─e→[Gc]─[k]──uc─→+◯─u─→[Gp]─yp─→+◯──┬──→ y
          -↑                           dm↓+        │
           └──yf─────[Gf]←───ym────────◯+←───────┘

      Open Loop Transfer Function:  T(s) = Gc*Gp*Gf
  '''

  def __init__(self, Gp, Gc=None, Gf=None, quantity_PZs = 4):
    """Gp: plant transfer function (Python Control Package);
       Gc (optional): controller transfer function (Python Control Package)
       Gf (optional): measurement filter transfer function
       Gp, Gc e Gf shall be of the same sample time Ts.  """
    from scipy.signal import tf2zpk
    from numpy import array
    from ipywidgets import Tab
    from IPython.display import display
    from control import tf, summing_junction

    #IPython.display.clear_output()
    self.dt = Gp.dt
    self.ContinuousTime = True if (self.dt is None or self.dt == 0) else False
    self.DiscreteTime = not self.ContinuousTime
    self.Gp = ControllerSISOApp(Gp.num, Gp.den, self.dt, name = 'Gp', inputs = ['u'], outputs = ['yp'])
    self.OLTF = tf(1,1,self.dt)
    self.PhaseMargin, self.GainMargin = 0,0
    self.rootsVect = []
    self.fNyquistHz = 1e6 if self.ContinuousTime else 0.5/self.dt;
    if Gc == None:  self.Gc = ControllerSISOApp([1],[1], self.dt, name = 'Gc', inputs = ['e'], outputs = ['uc'], start_configWgt=True)
    else:
        conditions = [self.ContinuousTime, Gc.dt==Gp.dt]
        assert any(conditions) ,  'Gc.dt should be equal to Gp.dt'
        self.Gc = ControllerSISOApp(Gc.num, Gc.den,0, name = 'Gc', inputs = ['e'], outputs = ['uc'],
                                start_configWgt=True, quantity_PZs = max(len(Gc.poles())+len(Gc.zeros()),4) )
    if Gf == None: self.Gf = tf([1],[1], self.dt, name = 'Gf', inputs = ['ym'], outputs = ['yf'])
    else:
        conditions = [self.ContinuousTime, Gf.dt==Gp.dt]
        assert any(conditions) ,  'Gf.dt should be equal to Gp.dt'
        self.Gf = tf(Gf.num, Gf.den, Gf.dt, name = 'Gf', inputs = ['ym'], outputs = ['yf'])

    #Defição do sistema em malha fechada:
    self.Sr = summing_junction(inputs = ['r', '-yf'], output = 'e', name = 'Sr');
    self.Sdu = summing_junction(inputs = ['uc', 'du'], output = 'u', name = 'Sdu');
    self.Sdy = summing_junction(inputs = ['yp','dy'], output = 'y', name = 'Sdy');
    self.Sdm = summing_junction(inputs = ['y', 'dm'], output = 'ym', name = 'Sdm');
    self.Gcgain = tf([self.Gc.Kdcgain],[1], self.dt, name = 'gain', inputs = ['uc1'], outputs = ['uc']);
    if self.DiscreteTime:
        self.Sr.dt, self.Sdu.dt, self.Sdy.dt, self.Sdm.dt = self.dt, self.dt, self.dt, self.dt
    self.Gp.name, self.Gp.input_labels, self.Gp.output_labels = 'Gp', ['u'], ['yp']
    self.Gc.name, self.Gc.input_labels, self.Gc.output_labels = 'Gc',  ['e'], ['uc1']
    self.Gf.name, self.Gf.input_labels, self.Gf.output_labels = 'Gf', ['ym'], ['yf']

    #Insert Widgets:
    self.CtrAnWgt = ControlAnalysisWidget(tmax_s = 10, dt_s = 1e-3 if self.ContinuousTime else self.dt, freq0_Hz = 1)
    self.UserApp = Tab([self.Gc.Appwidget, self.CtrAnWgt.RequirementsTab,self.CtrAnWgt.DisturbSimulationTab])
    self.UserApp.set_title(0,'Controller settings')
    self.UserApp.set_title(1,'Requirements')
    self.UserApp.set_title(2,'Disturbance simulation')

    #Define events:
    self.Gc.CgainInDBInteract.observe(self.updateGcgainAndBokeh,'value')
    for q in range(len(self.Gc.PZwidgets)):
        self.Gc.PZwidgets[q].SETwgt.observe(self.updateTFAndBokeh, 'button_style')
    self.Gc.SETBox[0].observe(self.updateTFAndBokeh, 'button_style')
    self.CtrAnWgt.OShotIn.observe(self.updateRequirements,'value')
    self.CtrAnWgt.RTimeIn.observe(self.updateRequirements,'value')
    self.CtrAnWgt.STimeIn.observe(self.updateRequirements,'value')
    self.CtrAnWgt.SETAllButton.observe(self.updateDistResponse, 'button_style')

    #show screen:
    self.buildBokehFigs()
    asdfs = display(self.UserApp)
    bokeh.io.output_notebook()
    self.Bknb_handle = bokeh.io.show(self.Bkgrid, notebook_handle=True)
    self.updateTFAndBokeh(0)
    #self.updateBokeh()

  #def printController(self,b):
  #  from control import db2mag
  #  numstr, denstr = '', ''
  #  Kstr = f'{db2mag(self.CgainInDBInteract.value):.4f}'
  #  for pz in self.PolesAndZeros:
  #    numstr = numstr + pz.printNumOrDen('num')
  #    denstr = denstr + pz.printNumOrDen('den')
  #  if len(numstr)<1: numstr = '1'
  #  if len(denstr)<1: denstr = '1'
  #  return str(numstr), str(denstr), Kstr

  def buildBokehFigs(self):
    from bokeh.models import ColumnDataSource,Span,Band,Label, Range1d
    from bokeh.plotting import figure as BkFig
    #BOKEH FIGURES:
    #Vector data:
    self.bodesource = ColumnDataSource( data={'omega':[], 'freqHz': [],
              'magdBT':[],'magT':[],'magdBG':[],'magG': [],'angT':[],'angG':[]})
    self.gpbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.gzbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.cpbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.czbodesource = ColumnDataSource(data ={'fHz':[],'magdB':[],'angdeg':[]})
    self.PM_GMsource = ColumnDataSource( data = {'PMfcHz': [1.,1.],'GMfHz':[2.,2.],  'ylimsmag':[-200,200], 'ylimsang':[-720,720] })
    self.rlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.gprlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.gzrlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.cprlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.czrlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.krlocussource = ColumnDataSource(data={'x':[],'y':[],'K':[]})
    self.stepsource = ColumnDataSource(data={'t_s':[],'stepRYmf':[],'stepUYma':[],'stepRUmf':[]})
    self.tRespsource = ColumnDataSource(data=self.CtrAnWgt.waveVec_dict)

    #Shadows:
    MAX_OVERSHOT = 0.01*self.CtrAnWgt.OShotIn.value + 1
    MAX_RISE_TIME, MAX_SETTLING_TIME = self.CtrAnWgt.RTimeIn.value, self.CtrAnWgt.STimeIn.value
    self.shadowsource = ColumnDataSource(
        data={'x_s': [0,1e4],     'ylow': [-1e4,1e4],  'yup': [1e4,1e4],
            'xn_z': [-2,-1], 'xp_z': [1,2] , 'zero':[0,0],
            'overshot':[MAX_OVERSHOT, MAX_OVERSHOT],
            'risetime':[MAX_RISE_TIME,1e4] , 'riselevel':[0.9,0.9],
            'settlingtime':[MAX_SETTLING_TIME,1e4],
            'setlevel1':[0.98,0.98], 'setlevel2':[1.02,1.02]  } )
    _thetaZ = np.linspace(0,np.pi,100)
    _costh, _sinthN, _sinth = np.cos(_thetaZ), -np.sin(_thetaZ), np.sin(_thetaZ)
    self.shadowZsource=ColumnDataSource( data = {'x_z':_costh, 'ylow_z':_sinthN, 'yup_z':_sinth,  'ylow': 100*[-1e4], 'yup': 100*[1e4]})

    self.shadows = {
    'rloc_s': Band(base='x_s', lower='ylow', upper='yup', level='underlay', source=self.shadowsource, fill_color='lightgrey', line_color='black'),
    'rloc_z1': Band(base='xn_z', lower=-1e3, upper=1e3, level='underlay',   source=self.shadowZsource, fill_color='lightgrey'),
    'rloc_z2': Band(base='xp_z', lower='ylow', upper='yup', level='underlay',  source=self.shadowZsource, fill_color='lightgrey'),
    'rloc_z3': Band(base='x_z', lower='ylow', upper='ylow_z', level='underlay',  source=self.shadowZsource,fill_color='lightgrey',line_color='black'),
    'rloc_z4': Band(base='x_z', lower='yup_z', upper='yup', level='underlay',  source=self.shadowZsource,fill_color='lightgrey',line_color='black'),
    'ovsht': Band(base='x_s', lower='overshot', upper='yup', level='underlay',  source=self.shadowsource,fill_color='lightgrey', visible=True),
    'riset': Band(base='risetime', lower='ylow', upper='riselevel',  level='underlay', source=self.shadowsource,fill_color='lightgrey'),
    'sett1': Band(base='settlingtime', lower='riselevel', upper='setlevel1',  level='underlay', source=self.shadowsource,fill_color='lightgrey'),
    'sett2': Band(base='settlingtime', lower='setlevel2', upper='overshot', level='underlay', source=self.shadowsource,fill_color='lightgrey') }

    _TTS_BD1 = [('sys',"$name"),("f","$x Hz"),("mag","$y dB")]
    _TTS_BD2 = [('sys',"$name"),("f","$x Hz"),("ang","$y°")]
    _TTS_RLOC= [("real","@x"),("imag","@y"),('K','@K{0.00 a}')]
    _TTS_TRESP = [('signal', "$name"), ("t", "$x s"), ("value", "$y") ]
    W = 450
    H = 300
    self.figMag = BkFig(title="Bode Magnitude", height=H, width=W,  toolbar_location="above", tooltips = _TTS_BD1, x_axis_type="log", x_axis_label='f (Hz)', y_axis_label='mag (dB)')
    self.figAng =  BkFig(title="Bode Angle", height=H, width=W, toolbar_location="above", tooltips = _TTS_BD2, x_axis_type="log",  x_axis_label='f (Hz)', y_axis_label='ang (°)')
    self.figAng.x_range  = self.figMag.x_range   #same axis
    self.figAng.yaxis.ticker = np.linspace(-720,720,17)
    self.figRLoc = BkFig(title="Root Locus", height=H, width=W, toolbar_location="above", tooltips = _TTS_RLOC, x_range=Range1d(-1.5, 1.5), y_range=Range1d(-1.5, 1.5),x_axis_label='real', y_axis_label='imag')
    #self.figRLoc.hover.line_policy = 'interp'
    self.figTResp = BkFig(title="Step Response", height=H, width=W,   toolbar_location="above", tooltips = _TTS_TRESP, x_axis_label='time (s)', y_axis_label='y')
    self.figTResp2= BkFig(title="Disturbance Simulation", height=H, width=2*W,  toolbar_location="above", tooltips = _TTS_TRESP, x_axis_label='time (s)', y_axis_label='y, r, dy, dm')

    self.Bkgrid = bokeh.layouts.layout([[self.figMag, self.figRLoc], [self.figAng, self.figTResp],  self.figTResp2])


    if self.ContinuousTime:   #continuous time
      self.figRLoc.add_layout(self.shadows['rloc_s'])
    else:                 #discrete time
        for strkey in ['rloc_z1', 'rloc_z2', 'rloc_z3', 'rloc_z4']:
          self.figRLoc.add_layout(self.shadows[strkey])
        self.Nyquistlimits = Span(location=0.5/self.dt, dimension='height', line_color='black',  line_dash='dotted', line_width=1)
        self.figMag.add_layout(self.Nyquistlimits)
        self.figAng.add_layout(self.Nyquistlimits)
    for strkey in ['ovsht', 'riset', 'sett1', 'sett2']:
      self.figTResp.add_layout(self.shadows[strkey])
      #self.figTResp2.add_layout(self.shadows[strkey])


    #Bode Diagram:
    bodemagT=self.figMag.line(x='freqHz', y='magdBT',color="blue",line_width=1.5,  alpha=0.8,name='|T(s)|',legend_label='T(s)',source=self.bodesource)
    bodemagG=self.figMag.line(x='freqHz', y='magdBG',color="green",line_width=1.5,  alpha=0.8,name='|Gp(s)|',line_dash='dashed',  legend_label='Gp(s)',source=self.bodesource)
    bodeangT=self.figAng.line(x='freqHz', y='angT', color="blue", line_width=1.5,  alpha=0.8, name='∡T(s)', source=self.bodesource)
    bodeangG=self.figAng.line(x='freqHz', y='angG',color="green", line_width=1.5,   alpha=0.8,name='∡Gp(s)',line_dash='dashed',source=self.bodesource)
    bodeGpmag = self.figMag.scatter( x='fHz',y='magdB',marker='x',line_color='blue', size=10,  name='Gp poles', source = self.gpbodesource)
    bodeGpang=self.figAng.scatter(x='fHz',y='angdeg',marker='x',line_color='blue', size=10,  name='Gp poles angle', source = self.gpbodesource)
    bodeGzmag = self.figMag.scatter(x='fHz',y='magdB',marker='circle',line_color='blue',size=8,  name='Gp zeros',fill_color=None, source = self.gzbodesource)
    bodeGzang=self.figAng.scatter(x='fHz',y='angdeg',marker='circle',line_color='blue',size=8, name='Gp zeros angle', fill_color=None,source = self.gzbodesource)
    bodeCpmag = self.figMag.scatter(x='fHz',y='magdB',marker='x',line_color='red',size=10,  name='C poles', source = self.cpbodesource)
    bodeCpang=self.figAng.scatter(x='fHz',y='angdeg',marker='x',line_color='red',size=10,   name='C poles angle', source = self.cpbodesource)
    bodeCzmag = self.figMag.scatter(x='fHz',y='magdB',marker='circle',line_color='red',size=8,  name='C zeros', fill_color=None, source = self.czbodesource)
    bodeCzang=self.figAng.scatter(x='fHz',y='angdeg',marker='circle',line_color='red',size=8, name='C zeros angle',fill_color=None, source = self.czbodesource)
    self.GMSpan = Span(location=1, dimension='height', line_color='black', line_dash='dotted', line_width=1)
    self.PMSpan = Span(location=1, dimension='height', line_color='black', line_dash='dotted', line_width=1)
    self.PMtxt = Label(x=5, y=5, x_units='screen', y_units='screen',  text=' ',  border_line_color=None,  background_fill_color='white',text_font_size = '11px')
    self.GMtxt = Label(x=5, y=20, x_units='screen', y_units='screen', text=' ', border_line_color=None,  background_fill_color='white',text_font_size = '11px')
    self.figMag.add_layout(self.GMSpan), self.figAng.add_layout(self.GMSpan)
    self.figMag.add_layout(self.PMSpan), self.figAng.add_layout(self.PMSpan)
    self.figAng.add_layout(self.PMtxt), self.figAng.add_layout(self.GMtxt)


    #Root Locus:
    rlocusline = self.figRLoc.scatter(x='x',y='y',marker='dot',color='blue', name='rlocus', source = self.rlocussource)
    rlocusGpoles = self.figRLoc.scatter(x='x',y='y',marker='x',color='blue', size=10,  name='Gp pole', source = self.gprlocussource)
    rlocusGzeros = self.figRLoc.scatter(x='x',y='y',marker='circle',line_color='blue',size=8,  name='Gp zero', fill_color=None, source = self.gzrlocussource)
    rlocusCpoles = self.figRLoc.scatter(x='x',y='y',marker='x',color='red', size=10, name='C pole', source = self.cprlocussource)
    rlocusCzeros = self.figRLoc.scatter(x='x',y='y',marker='circle',line_color='red',size=8,  name='C zero', fill_color=None, source = self.czrlocussource)
    rlocusMF = self.figRLoc.scatter(x='x',y='y',marker='square', line_color='deeppink',size=8, name='K', fill_color='deeppink', source = self.krlocussource)
    rlocuslinehv = self.figRLoc.line(x='x',y='y',line_alpha=0, name='rlocus2', source = self.rlocussource)
    self.figRLoc.hover.renderers=[rlocuslinehv, rlocusGpoles, rlocusGzeros, rlocusCpoles, rlocusCzeros, rlocusMF]
    #self.figRLoc.hover.mode='mouse'
    #self.figRLoc.hover.line_policy='next'
    #self.figRLoc.hover.point_policy='snap_to_data'
    self.Stabilitytxt = Label(x=10, y=200, x_units='screen', y_units='screen',  text=' ', border_line_color=None,  background_fill_color='white',text_font_size = '11px')
    self.figRLoc.add_layout(self.Stabilitytxt)

    #Step response:
    self.figTResp.extra_y_ranges = {'u_range': bokeh.models.Range1d()}
    self.figTResp.add_layout(bokeh.models.LinearAxis(y_range_name="u_range",  axis_label='u'), 'right')
    self.figTResp.y_range = bokeh.models.Range1d(start = -0.1, end = 1.4)
    #add_graf = self.figTResp.line if self.dt in [None, 0.0] else self.figTResp.dot
    if self.ContinuousTime:
        stepR2Y=self.figTResp.line(x='t_s', y='stepRYmf',color="blue",line_width=1.5, name='y', legend_label='y (closed loop)',  source=self.stepsource)
        stepU2Y=self.figTResp.line(x='t_s', y='stepUYma',color="green",  legend_label='y (open loop)', line_dash='dashed', line_width=1.0, name='y (ol)',source=self.stepsource, visible=False)
        stepR2U=self.figTResp.line(x='t_s', y='stepRUmf',color="red", line_width=1.0, name='u',legend_label='u (closed loop)', line_dash='dashed', source=self.stepsource, y_range_name = 'u_range', visible=False)
    else:
        stepR2Y=self.figTResp.scatter(x='t_s', y='stepRYmf',marker='dot',color="blue", line_width=1.5, name='y',  size=15,  legend_label='y (closed loop)',  source=self.stepsource)
        stepU2Y=self.figTResp.scatter(x='t_s', y='stepUYma',marker='dot',color="green", size=15, legend_label='y (open loop)', line_width=1.0, name='y (ol)',source=self.stepsource, visible=False)
        stepR2U=self.figTResp.scatter(x='t_s', y='stepRUmf',marker='dot',color="red", size=15, line_width=1.0, name='u',legend_label='u (closed loop)', source=self.stepsource, y_range_name = 'u_range', visible=False)
    self.figTResp.legend.location = 'bottom_right'
    self.figTResp.legend.click_policy = 'hide'

    #Disturbances response:
    self.figTResp2.extra_y_ranges = {'u_range': bokeh.models.Range1d()}
    self.figTResp2.add_layout(bokeh.models.LinearAxis(y_range_name="u_range",
                                                     axis_label='u, du'), 'right')
    self.figTResp2.y_range = bokeh.models.Range1d(start = -0.1, end = 1.4)
    if self.ContinuousTime:
        tRespY=self.figTResp2.line(x='t_s', y='y',color="blue", line_width=1.5, name='y', legend_label='y',  source=self.tRespsource)
        tRespU=self.figTResp2.line(x='t_s', y='u',color="red", legend_label='u', line_width=1.5,  name='u',source=self.tRespsource, y_range_name = 'u_range', visible=False)
        tRespDU=self.figTResp2.line(x='t_s', y='du',color="indianred", line_width=1.0, name='du',legend_label='du',  line_dash='dashed', source=self.tRespsource, y_range_name = 'u_range', visible=False)
        tRespR=self.figTResp2.line(x='t_s', y='r',color="green", line_width=1.0, name='r',legend_label='r', line_dash='dashed', source=self.tRespsource)
        tRespDY=self.figTResp2.line(x='t_s', y='dy',color="deepskyblue", line_width=1.0, name='dy',legend_label='dy',  line_dash='dashed', source=self.tRespsource, visible=False)
        tRespDM=self.figTResp2.line(x='t_s', y='dm',color="lime", line_width=1.0, name='dm',legend_label='dm', line_dash='dashed', source=self.tRespsource, visible=False)
    else:
        tRespY=self.figTResp2.scatter(x='t_s', y='y',marker='dot',color="blue", line_width=1.5, size=15, name='y', legend_label='y',  source=self.tRespsource)
        tRespU=self.figTResp2.scatter(x='t_s', y='u',marker='dot',color="red", legend_label='u', line_width=1.5, size=15, name='u',source=self.tRespsource, y_range_name = 'u_range', visible=False)
        tRespDU=self.figTResp2.scatter(x='t_s', y='du',marker='dot',color="indianred", line_width=1.0, name='du',legend_label='du', size=15, line_dash='dashed', source=self.tRespsource, y_range_name = 'u_range', visible=False)
        tRespR=self.figTResp2.scatter(x='t_s', y='r',marker='dot',color="green", line_width=1.0, name='r',legend_label='r',size=15, line_dash='dashed', source=self.tRespsource)
        tRespDY=self.figTResp2.scatter(x='t_s', y='dy',marker='dot',color="deepskyblue", line_width=1.0, name='dy',legend_label='dy', size=15,line_dash='dashed', source=self.tRespsource, visible=False)
        tRespDM=self.figTResp2.scatter(x='t_s', y='dm',marker='dot',color="lime", line_width=1.0, name='dm',legend_label='dm',  size=15, line_dash='dashed', source=self.tRespsource, visible=False)
    self.figTResp2.legend.location = 'bottom_right'
    self.figTResp2.legend.click_policy = 'hide'

  def updateTFAndBokeh(self,b):
    self.updateTransferFunction()
    self.createBode()
    self.createRLocus()
    self.updateBokeh()

  def updateGcgainAndBokeh(self,b):
    self.updateTransferFunction();
    self.updateBokeh()

  def updateBokeh(self):
    self.updateBodeData()
    self.updateRLocusData()
    self.updateStepResponse()
    self.updateDistResponse({'new':'success'})
    bokeh.io.push_notebook(handle = self.Bknb_handle);

  def updateTransferFunction(self):
    from control import interconnect, minimal_realization, tf
    self.Gcgain = tf([self.Gc.Kdcgain],[1], self.dt, name = 'gain', inputs = ['uc1'], outputs = ['uc']);
    self.OLTF = minimal_realization(self.Gc*self.Gcgain*self.Gp*self.Gf,tol=1e-6,verbose=False)
    self.Gc.name, self.Gc.input_labels, self.Gc.output_labels = 'Gc',  ['e'], ['uc1']
    self.sysMF = interconnect([self.Gc, self.Gcgain, self.Gp, self.Gf, self.Sr, self.Sdu, self.Sdy, self.Sdm],
                              inputs = ['r','du','dy','dm'] ,  outputs = ['y', 'uc']);

  def _clamp_fNyquist(self,SPoles_SZeros):
    return np.clip(np.abs(SPoles_SZeros), 0, 2*np.pi*self.fNyquistHz)

  def createBode(self):
    '''Creates the plots for Bode Diagram '''
    from control.matlab import mag2db
    from control import frequency_response
    R2D,  W2F = 180/np.pi,  1/(2*np.pi)
    #Definition of the frequency range:
    freqs = np.abs(np.concatenate([
        np.atleast_1d(self.Gp.SPoles), np.atleast_1d(self.Gp.SZeros),
        np.atleast_1d(self.Gc.SPoles), np.atleast_1d(self.Gc.SZeros)]))
    freqs_nz = freqs[freqs > 1e-9]   #frequêncies different than zero
    omega_min = 0.1 * np.min(freqs_nz) if len(freqs_nz) > 0 else 1e-3
    omega_max =  10*np.max(freqs_nz) if self.ContinuousTime else 0.999*np.pi/self.dt
    omega_max = max(1e-2, omega_max)
    omega_range =np.logspace(np.log10(omega_min), np.log10(omega_max), 1000)
    magT,phiT,omega = frequency_response(self.OLTF, omega = omega_range)
    magG,phiG,_ = frequency_response(self.Gp, omega)
    magdbG, magdbT = mag2db(magG), mag2db(magT)
    self.bodesource.data={'omega':omega, 'freqHz':(omega/(2*np.pi)),
                         'magdBG':magdbG, 'magG':magG, 'angG':phiG*R2D,
                         'magdBT':magdbT, 'magT':magT, 'angT':phiT*R2D}
    self.updatePMGM()
    #func1 = np.abs if self.dt in [None, 0.0] else self.d2c_clampAtNyquistFreq
    dict1 = {'GpP': [self.Gp.SPoles,self.gpbodesource],
             'GpZ': [self.Gp.SZeros,self.gzbodesource],
             'GcP': [self.Gc.SPoles,self.cpbodesource],
             'GcZ': [self.Gc.SZeros,self.czbodesource]}
    for key1 in ['GpP','GpZ','GcP','GcZ']:
      pORz = list(filter(lambda x: x>1e-5, self._clamp_fNyquist(dict1[key1][0])))
      magdB, phideg, fHz = [], [], []
      if pORz:
        mag,phi,omega = frequency_response(self.OLTF, np.abs(pORz))
        magdB, phideg, fHz = mag2db(mag), phi*R2D, (omega*W2F)
        for q in range(len(phideg)):
          if phideg[q] > 90: phideg[q] = phideg[q]-360;
      dict1[key1][1].data={'fHz':list(fHz),'magdB':list(magdB),'angdeg':list(phideg)}

  def updateBodeData(self):
    from control.matlab import mag2db
    def sum_constant_to_list(data_dict, list_key, constant):
          data_dict[list_key] = list(np.array(data_dict[list_key])+constant)
    dmagdB, dmag = mag2db(self.Gc.dKdcgain), self.Gc.dKdcgain
    for pz in [self.gpbodesource,self.gzbodesource,self.cpbodesource,self.czbodesource]:
        sum_constant_to_list(pz.data,'magdB', dmagdB)
    sum_constant_to_list(self.bodesource.data,'magdBT', dmagdB )
    self.bodesource.data['magT']=list(dmag*np.array(self.bodesource.data['magT']))
    self.updatePMGM()

  def updatePMGM(self):
    from control.matlab import margin, mag2db
    R2D,  W2F, F2W = 180/np.pi,  1/(2*np.pi),  2*np.pi
    self.GainMargin,self.PhaseMargin,wg,wc = margin(self.OLTF)
    if np.isnan(wg): wg = self.fNyquistHz*F2W
    if np.isnan(wc): wc = self.fNyquistHz*F2W
    self.PMSpan.location = wc*W2F
    self.GMSpan.location = wg*W2F
    if str(self.GainMargin) == 'inf':  self.GMtxt.text = 'GM: inf'
    else: self.GMtxt.text = f'GM:{mag2db(self.GainMargin):.1f} dB'
    if str(self.PhaseMargin) == 'inf': self.PMtxt.text = 'PM: inf'
    else: self.PMtxt.text = f'PM: {self.PhaseMargin:.1f}°'

  def createRLocus(self):
    from numpy import real, imag, array
    from control import root_locus_map
    GcdB, Gcmag = self.Gc.CgainInDBInteract.value, self.Gc.Kdcgain
    Kgp, Kgz = np.zeros(len(self.Gp.Poles)), np.inf*np.ones(len(self.Gp.Zeros))
    Kcp, Kcz = np.zeros(len(self.Gc.Poles)), np.inf*np.ones(len(self.Gc.Zeros))
    pzdata = root_locus_map(self.OLTF/self.Gc.Kdcgain , gains = self.Gc.kvect )
    self.rootsVec = pzdata.loci
    Krlocus,  cols = array(self.Gc.kvect),  self.rootsVec.shape[1]-1
    for x in range(cols):  Krlocus = np.column_stack((Krlocus,self.Gc.kvect))
    self.rlocussource.data = {'x': (real(self.rootsVec)).flatten(),
                              'y': (imag(self.rootsVec)).flatten(),
                              'K': Krlocus.flatten()}

    self.gprlocussource.data = {'x':real(self.Gp.Poles),'y':imag(self.Gp.Poles),'K':Kgp}
    self.gzrlocussource.data = {'x':real(self.Gp.Zeros),'y':imag(self.Gp.Zeros),'K':Kgz}
    self.cprlocussource.data = {'x':real(self.Gc.Poles),'y':imag(self.Gc.Poles),'K':Kcp}
    self.czrlocussource.data = {'x':real(self.Gc.Zeros),'y':imag(self.Gc.Zeros),'K':Kcz}
    #print(self.czrlocussource.data)
    self.updateRLocusData()

    if self.rootsVec.size>0 and self.ContinuousTime:
      xrangemin, xrangemax = np.min(real(self.rootsVec)), np.max(real(self.rootsVec))
      if np.abs(xrangemax-xrangemin)<2:
        self.figRLoc.x_range.update(start=xrangemin-1, end=xrangemax+1)
      yrangemin, yrangemax = np.min(imag(self.rootsVec)), np.max(imag(self.rootsVec))
      if np.abs(yrangemax-yrangemin)<2:
        self.figRLoc.y_range.update(start=yrangemin-1, end=yrangemax+1)
    if self.DiscreteTime:
      self.figRLoc.x_range.update(start=-1.5, end=1.5)
      self.figRLoc.y_range.update(start=-1.5, end=1.5)


  def updateRLocusData(self):
    Kindex = int(len(self.Gc.kvect)*(self.Gc.CgainInDBInteract.value-self.Gc.kvectdB[0])
                                       /(self.Gc.kvectdB[-1]-self.Gc.kvectdB[0]))-1
    x,y = np.real(self.rootsVec[Kindex-1]), np.imag(self.rootsVec[Kindex-1])    #problemas com tamanho do vetor!!
    K = self.Gc.kvect[Kindex-1]*np.ones(len(list(x)))
    self.krlocussource.data = {'x': x , 'y': y, 'K':K}
    unstable = (x>0) if self.dt in [None, 0.0] else  ((x*x+y*y)>1)
    if any(unstable): self.Stabilitytxt.text = 'Unstable Loop'
    else:             self.Stabilitytxt.text = 'Stable Loop'

  def updateStepResponse(self):
    from control import step_response
    p_dom = np.abs(np.real(self.sysMF[0,0].poles()))
    wp_dom = p_dom if self.ContinuousTime else -np.log(np.clip(p_dom, 1e-12, 0.9999))/self.dt
    tau5_Gmf = np.abs(5/np.clip(np.min(wp_dom),1e-5,1e6)) #5 constantes de tempo
    Npoints = 2000
    dt = tau5_Gmf/2000 if self.ContinuousTime else self.dt
    tfinal = tau5_Gmf if (self.dt is None or self.dt == 0.0) else max(tau5_Gmf, 20*dt)
    tvec, yu = step_response(self.sysMF, T = np.arange(0, tfinal, dt), input=0)
    ymf, umf = np.atleast_1d(yu[0,:][0]), np.atleast_1d(yu[1,:][0])
    _,yma = step_response(self.Gp, T=tvec)
    tvec, yma = np.atleast_1d(tvec), np.atleast_1d(yma)
    self.stepsource.data={'t_s':tvec,'stepRYmf':ymf,'stepUYma':yma,'stepRUmf':umf }
    #print('tvec:',tvec, '\n ymf: ',ymf, '\n umf: ',umf, '\n yma:', yma)
    umin, umax = 0.9*float(np.min(umf)), 1.2*float(np.max(umf))
    self.figTResp.extra_y_ranges['u_range'].update(start=umin, end=umax)

  def updateRequirements(self,b):
    max_overshot = 0.01*self.CtrAnWgt.OShotIn.value+1
    self.shadowsource.data['overshot'] = [max_overshot, max_overshot]
    self.shadowsource.data['risetime'] = [self.CtrAnWgt.RTimeIn.value, 1e4]
    self.shadowsource.data['settlingtime'] = [self.CtrAnWgt.STimeIn.value, 1e4]
    bokeh.io.push_notebook(handle = self.Bknb_handle)


  def updateDistResponse(self,button_style):
    if button_style['new']=='success' and len(self.CtrAnWgt.waveVec_dict['t_s']>90):
      from control.matlab import lsim
      t,r,du,dy,dm = map(self.CtrAnWgt.waveVec_dict.get, ('t_s','r','du','dy','dm'))
      y, u = np.zeros_like(t), np.zeros_like(t)
      yu,_,_ = lsim(self.sysMF, U=np.column_stack((r,du,dy,dm)), T=t)
      ymf, umf = yu[:,0], yu[:,1]
      self.tRespsource.data = {'t_s':t,'r':r,'du':du,'dy':dy,'dm':dm,'y':ymf,'u':umf}
      ymin, ymax, umin, umax = float(np.min(ymf)), float(np.max(ymf)), float(np.min(umf)), float(np.max(umf))
      umin = min(umin, np.min(du))
      umax = max(umin, np.max(du))
      self.figTResp2.y_range.update(start = ymin - 0.05*(ymax-ymin),
                                    end = ymax + 0.05*(ymax-ymin))
      self.figTResp2.extra_y_ranges['u_range'].update(start = umin - 0.05*(umax-umin),
                                                      end = umax + 0.05*(umax-umin))
      bokeh.io.push_notebook(handle = self.Bknb_handle)
