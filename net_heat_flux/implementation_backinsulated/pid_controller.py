"""
This script defines a PID (Proportional, Integral, Derivative) controller function.

It is used to calculate the new value of u(t) = MV (the manipulated variable).
It uses the current value of the measured variable, the set point (desired value of the measured variable) and the error history
"""

def PID(Input, Setpoint, lastErr, lastInput, errSum, timeChange, kp, ki, kd):
    pass
    """
    Uses a PID algorithm to calculate the IHF based on target and current nhf
    Output: IHF. Input: NHF: Setpoint: Target NHF.
    See: http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/
    """

    # Compute all the working error variables
    error = Setpoint - Input
    errSum += error * timeChange
    # dERr ia according to the initial design, but later dInput is used
    dErr = (error - lastErr) / timeChange

    # Compute the Output
    Output = kp * error + ki * errSum - kd * dErr
    
    # physical limits of the FPA
    if Output < 0:
        Output = 0
    elif Output > 55000:
        Output = 55000

    return Output, error, errSum