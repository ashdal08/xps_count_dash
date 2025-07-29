# -*- coding: utf-8 -*-
import random
import threading
import time

import numpy as np


class DAC1_8:
    """
    16-bit DAC Feedback command for DAC1

    Controls DAC1 in 16-bit mode.

    Value: 0-65535

    >>> d.getFeedback( u6.DAC1_16( Value ) )
    [ None ]
    """

    def __init__(self, Value):
        return "dac1"


class DAC0_8:
    """
    16-bit DAC Feedback command for DAC1

    Controls DAC1 in 16-bit mode.

    Value: 0-65535

    >>> d.getFeedback( u6.DAC1_16( Value ) )
    [ None ]
    """

    def __init__(self, Value):
        return "dac0"


class DAC1_16:
    """
    16-bit DAC Feedback command for DAC1

    Controls DAC1 in 16-bit mode.

    Value: 0-65535

    >>> d.getFeedback( u6.DAC1_16( Value ) )
    [ None ]
    """

    def __init__(self, Value):
        return "dac1"


class DAC0_16:
    """
    16-bit DAC Feedback command for DAC1

    Controls DAC1 in 16-bit mode.

    Value: 0-65535

    >>> d.getFeedback( u6.DAC1_16( Value ) )
    [ None ]
    """

    def __init__(self, Value):
        return "dac0"


class Counter0:
    """
    Counter0 Feedback command

    Reads hardware counter0, optionally resetting it

    Reset: True ( or 1 ) = Reset, False ( or 0 ) = Don't Reset

    Returns the current count from the counter if enabled.  If reset,
    this is the value before the reset.

    >>> d.getFeedback(u6.Counter0(Reset = False))
    [ 2183 ]
    """

    def __init__(self, Reset=False):
        return "counter_zero" if Reset else "counter"


class U6:
    counter: list
    __counter_thread: threading.Thread
    __dac_1: float
    __dac_0: float
    __D_PIN_STATES: list = [0, 0, 0, 0, 0, 0, 0, 0]

    def __init__(self):
        pass

    def getCalibrationData(self):
        pass

    def getFeedback(self, *commandlist) -> list:
        if commandlist[0] == "counter":
            return self.counter
        elif commandlist[0] == "dac0":
            # return self.__dac_0
            return []
        elif commandlist[0] == "dac1":
            # return 3 + (random.random() - 0.5) * 0.01
            return []
        else:  # when selector == 'counter_zero'
            temp = self.counter
            self.counter = [0, 0]
            return temp

    def configIO(
        self,
        NumberTimersEnabled=None,
        EnableCounter1=None,
        EnableCounter0=None,
        TimerCounterPinOffset=None,
        EnableUART=None,
    ):
        if EnableCounter0:
            self.counter = [0, 0]
            self.__counter_thread = threading.Thread(target=self.__dummy_counter_thread, daemon=True)
            self.__counter_thread.start()
        pass

    def getAIN(self, positiveChannel: int, resolutionIndex=0, gainIndex=0, settlingFactor=0, differential=False):
        if positiveChannel == 3:
            return 3 + (random.random() - 0.5) * 0.01
        elif positiveChannel == 0:
            return self.__dac_0
        else:
            return self.__dac_1

    def setDOState(self, ioNum: int, state: int = 1):
        self.__D_PIN_STATES[ioNum] = state

    def getDIState(self, ioNum: int):
        return self.__D_PIN_STATES[ioNum]

    def configU6(self):
        pass

    def voltageToDACBits(self, volts, dacNumber=0, is16Bits=False):
        if dacNumber:
            self.__dac_1 = volts
        else:
            self.__dac_0 = volts
        return volts

    def spi(
        self,
        SPIBytes,
        AutoCS=True,
        DisableDirConfig=False,
        SPIMode="A",
        SPIClockFactor=0,
        CSPinNum=0,
        CLKPinNum=1,
        MISOPinNum=2,
        MOSIPinNum=3,
        CSPINNum=None,
    ):
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
                value += peak_height * np.exp(-((x - peak_center) ** 2) / (2 * random.randint(1, 50) ** 2))

            value += 0.2 * np.random.randn()
            self.counter[0] = value
            x += 1
            time.sleep(0.2)
