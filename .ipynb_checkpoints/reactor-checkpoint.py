#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:58:04 2024

@author: arjunmnanoj
"""
import numpy as np
import matplotlib.pyplot as plt


V = 100 #cm3
P = 1   #atm
T = 925 #K

# Feed has ethane in a steam diluent
# inlet partial pressure of ethane is 50 Torr and steam is 710 torr
Q = 35 # feed flow rate cm3



def rate_constants(T):
    A = np.array([1e17, 2e11, 3e14, 3.4e12, 1.6e13])
    E = np.array([356, 44, 165, 28, 0])
    R = 8.314
    k = A* np.exp(E/(R*T))
    return k


k = rate_constants(T)


def rxrate(v,x,p):
    N1, N2, N3, N4, N5, N6, N7, N8 = x

    Ntot = N1 + N2 + N3 + N4 + N5 + N6 + N7 + N8;
    Ctot = p.P/(p.R1*p.T);
    C1 = (N1/Ntot)*Ctot;
    C2 = (N2/Ntot)*Ctot;
    C4 = (N4/Ntot)*Ctot;
    C5 = (N5/Ntot)*Ctot;

    r1 = p*k[1]*C1;
    r2 = p*k[2]*C1*C2;
    r3 = p*k[3]*C4;
    r4 = p*k[4]*C1*C5;
    r5 = p*k[5]*C4*C5;

    dNdv = np.zeros(8, 1);

    dNdv[1] = -r1 - r2 - r4 + r5;
    dNdv[2] = 2*r1 - r2;
    dNdv[3] = r2;
    dNdv[4] = r2 - r3 + r4 - r5;
    dNdv[5] = r3  - r4 - r5;
    dNdv[6] = r3;
    dNdv[7] = r4;
    dNdv[8] = 0;
    
    return dNdv





Components_1 = [' C2H6 = 1,',' CH3 = 2,',' CH4 = 3,',' C2H5 = 4'];
Components_2 = [' H = 5,',' C2H4 = 6,', ' H2 = 7,',' H2O = 8'];

Ao = [1e17, 2e11, 3e14, 3.4e12, 1.6e13]';
Ea = [356000, 44000, 165000, 28000, 0]';
nu = [-1,2,0,0,0,0,0,0
      -1,-1,1,1,0,0,0,0
       0,0,0,-1,1,1,0,0
      -1,0,0,1,-1,0,1,0
       1,0,0,-1,-1,0,0,0];

R = 8.3144;  % Gas Constant (J/gmole-K)
R1 = 82.057; % Gas Constant (cc-atm/gmole-K)
T = 925; % Temp (K)
EXP = exp(-Ea/(R*T));
k = Ao.*EXP;
kp = (k(1)/(2*k(3)) + ((k(1)/(2*k(3)))^2 + ...
     ((k(1)*k(4))/(k(3)*k(5))))^0.5);
C1o = (50/760)/(82.057*T);  %gmole/cm3
C8o = (710/760)/(82.057*T);
Qf = 35.0;	%cc/sec
N1o = C1o*Qf;	%gmole/sec
N8o = C8o*Qf;
P = 1.0;	%atm
Initial = [N1o,0,0,0,0,0,0,N8o]';


v = [0:1:100]';
p = struct(); % Create structure to pass parameters to ode15s function
p.k = k;
p.T = T;
p.P = P;
p.R1 = R1;

opts = odeset ('AbsTol', sqrt (eps), 'RelTol', sqrt (eps));
[tsolver, solution] = ode15s(@(v,x) rxrate(v,x,p),v,Initial,opts);
answer = [v solution];
methane = solution(:,3);

ethane = solution(:,1);
ethylene = solution(:,6);
hydrogen = solution(:,7);
stable=[v ethane ethylene hydrogen methane];

temp = [v, solution];

save -ascii ethane_exact.dat temp;

if (~ strcmp (getenv ('OMIT_PLOTS'), 'true')) % PLOTTING
    subplot (2, 1, 1);
    plot (temp(:,1),[temp(:,7),temp(:,2)]);
    % TITLE ethane_exact

    subplot (2, 1, 2);
    plot (temp(:,1),[temp(:,4)]);
    % TITLE ethane_exact
end % PLOTTING