# Remote Operation of the Source
During longer measurements, for safety measures it is recommended to not be present around the X-ray source for the entire period. In addition, the life of the X-ray source decreases with longer usage. Hence it may be necessary to shutdown the X-ray source remotely, in case a measurement runs in to the night, or at a time when an operator is not permitted to be in the lab. This manual explains the connection setup and the steps for the setup of the instrument and the program to be able to shutdown the X-ray source remotely.

## Steps to follow to enable remote shutdown
1. Follow the instrument manual to ramp up the filament current and then switch to the minimum level of the emission current (0.5 as mentioned in "On the TX400 Control Unit" in case of normal use).
2. On the program in computer, click the "Remote Operate X-ray Source" button.
3. For the TX400, set the filament (Mg or Al) being used and select emission mode. **Selecting the correct filaments and modes is very important, else the filaments can break**.
4. In the *Current* field, enter the emission current currently seen in the LCD display of the TX400 unit
5. Click the *Send Remote Signal* button.
6. On the TX400 unit, change the switch from *Local* to *Remote*.
7. Now increase the emission current in the *Current* field in small increments, like 0.25 mA. After every increment, click on the *Send Remote Signal* button. You should notice that the emission current is increasing on the LCD display of the TX400 unit. Increment the emission current until the desired emission current is observed in the LCD display of the TX400 unit.
8. The instrument is now ready for measurements and is also setup for a remote shutdown.

## Steps to trigger the remote shutdown

**Only do the following steps if the instrument was initial setup for a remote shutdown as mentioned in the steps above.**

1. Reduce the emission current in the *Current* field of the program in decrements of 0.5 mA until 0 mA is reached. After every decrement click on the *Send Remote Signal* button.
2. When the emission current is down to 0 mA, the same should be displayed in the SL600 section of the program under *milliamperes*. It needn't be exactly 0 mA but slightly offset. For e.g. 0.29 mA.
3. If the *milliamperes* of the SL600 are down to 0 mA as mentioned, click the *Inhibit HV* button. This should tell the SL600 to power down the HV to the X-ray source. The *HV* display of the SL600 on the program should also show 0 kV.
4. Now the X-ray source has been remotely shutdown.
5. **DO NOT ATTEMPT TO START THE X-RAY SOURCE REMOTELY. STARTUP PROCEDURE SHOULD ALWAYS BE MADE LOCALLY AT THE INSTRUMENT.**

**Steps to bring the instrument back to local control**
1. Turn down the dials for the Filament current and emission current on the TX400 unit. Also switch the TX400 unit to filament mode. Turn down the dials for HV and milliamperes on SL600 unit.
2. On the TX400 unit switch back to local mode.
3. On the program, under SL600, click the *Inhibit HV* button.
4. The instrument is now back to fully local control and can now be started for operation again.