from Phidget22.Phidget import *
from Phidget22.Devices.VoltageRatioInput import *

ch = VoltageRatioInput()
ch.setChannel(0)
ch.openWaitForAttachment(1000)

#data interval is 10s
bridgeGain = -4858390.09050911
offset = -1.2844E-005
dataRate = 100

ch.setBridgeEnabled(True)
ch.setBridgeGain(bridgeGain)
ch.setDataRate(100)



ch.close()