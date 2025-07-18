# -*- coding: utf-8 -*-
import random
import threading
import time

import numpy as np

def DAC1_8(volt_bits):
    return 'dac1'

def DAC0_8(volt_bits):
    return 'dac0'

def Counter0(Reset: bool = False):
    return 'counter_zero' if Reset else 'counter'

class U6():
    counter : list
    __counter_thread : threading.Thread
    __dac_1 : float
    __dac_0 : float
    __D_PIN_STATES : list = [0, 0, 0, 0, 0, 0, 0, 0]
    
    def __init__(self):
        pass
    
    def getCalibrationData(self):
        pass
    
    def getFeedback(self, selector) -> list:
        if selector == 'counter':
            return self.counter
        elif selector == 'dac0':
            # return self.__dac_0
            return []
        elif selector == 'dac1':
            # return 3 + (random.random() - 0.5) * 0.01
            return []
        else: # when selector == 'counter_zero'
            temp = self.counter
            self.counter = [0,0]
            return temp
    
    def configIO(self, EnableCounter0 = False, EnableCounter1 = False, TimerCounterPinOffset = 6, FIOAnalog = 0, EIOAnalog = 0, EnableUART = False, TimerCounterConfig = 0, NumberOfTimersEnabled = 0, EnableCounter = False):
        if EnableCounter0:
            self.counter = [0, 0]
            self.__counter_thread = threading.Thread(target=self.__dummy_counter_thread, daemon=True)
            self.__counter_thread.start()
        pass
    
    def getAIN(self, channelNumber: int, resolutionIndex: int = 0, gainIndex: int = 0, differential = False, longSettle = False, quickSample = False):
        if channelNumber == 3:
            return 3 + (random.random() - 0.5) * 0.01
        elif channelNumber == 0:
            return self.__dac_0
        else:
            return self.__dac_1
        
    def setDOState(self, ioNum: int, state:int):
        self.__D_PIN_STATES[ioNum] = state

    def getDIState(self, ioNum: int):
        return self.__D_PIN_STATES[ioNum]
    
    def configU6(self):
        pass
    
    def voltageToDACBits(self, voltage, dacNumber: int):
        if dacNumber:
            self.__dac_1 = voltage
        else:
            self.__dac_0 = voltage
        return voltage
    
    def spi(self, SPIBytes, AutoCS=True, DisableDirConfig = False, SPIMode = 'A', SPIClockFactor = 0, CSPinNum = 0, CLKPinNum = 1, MISOPinNum = 2, MOSIPinNum = 3, CSPINNum = None):
        # Combine the 3 bytes into a 24-bit integer
        combined = (SPIBytes[0] << 16) | (SPIBytes[1] << 8) | SPIBytes[2]
        
        # Ignore the 2 MSBs (command bits)
        combined = combined & 0x3FFFFF
        
        # Shift right by 6 bits
        shifted = combined >> 6
        
        # Scale the value
        scaled_value = (shifted * self.__dac_1) / 65535
        
        self.__dac_0 = scaled_value
    
    def __dummy_counter_thread(self):
        while True:
            self.counter[0] += random.randint(1, 100)
            time.sleep(0.2)
            
    def __peak_generator(self):
        x = 0
        num_peaks = random.randint(1, 100)
        
        # Generate random peaks
        peak_centers = np.random.randint(0, 1000, num_peaks)
        peak_heights = np.random.uniform(0, 1000, num_peaks)
        
        while True:
            value = 0
            
            # Add peaks and noise
            for peak_center, peak_height in zip(peak_centers, peak_heights):
                value += peak_height * np.exp(-(x - peak_center)**2 / (2 * random.randint(1, 50)**2))
            
            value += 0.2 * np.random.randn()
            self.counter[0] = value
            x += 1
            time.sleep(0.2)