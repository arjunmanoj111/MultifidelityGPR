import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt



class PFR_ethane:
    def __init__(self, V, model='exact', P=1.0, T=925, Q = 35.0, feed=[50, 710]):
        """
        Feed has ethane in a steam diluent
        Inlet partial pressure of ethane is 50 Torr and steam is 710 torr
        """
        self.V = V
        self.P = P # atm
        self.T = T # Temp (K)
        self.Q = Q # cm3/sec
        self.R = 8.3144 # Gas Constant (J/gmole-K)
        self.R1 = 82.057 # Gas Constant (cm3-atm/gmole-K)
        self.P_eth = feed[0]
        self.P_steam = feed[1]
        self.model = model
        
    def rate_constants(self, T=None):
        """
        Rate Constants for a specific temperature
        """
        A = np.array([1e17, 2e11, 3e14, 3.4e12, 1.6e13])
        Ea = np.array([356000, 44000, 165000, 28000, 0])
        if T ==None:
            T = self.T
        k = A * np.exp(-Ea / (self.R * T))
        self.k = k
        self.kp = (k[0] / (2 * k[2]) + ((k[0] / (2 * k[2]))**2 + ((k[0] * k[3]) / (k[2] * k[4])))**0.5)
        
    def exact(self, v, x):
        """
        List of components
        0 - C2H6, 
        1 - CH3,
        2 - CH4,
        3 - C2H5,
        4 - H,
        5 - C2H4,
        6 - H2,
        7 - H2O
        """
        N1, N2, N3, N4, N5, N6, N7, N8 = x
        Ntot = sum(x)
        Ctot = self.P/(self.R1 * self.T)
        C1 = (N1/Ntot) * Ctot
        C2 = (N2/Ntot) * Ctot
        C4 = (N4/Ntot) * Ctot
        C5 = (N5/Ntot) * Ctot
        
        r1 = self.k[0] * C1
        r2 = self.k[1] * C1 * C2
        r3 = self.k[2] * C4
        r4 = self.k[3] * C1 * C5
        r5 = self.k[4] * C4 * C5
        
        dNdv = np.zeros(8)
        
        dNdv[0] = -r1 - r2 - r4 + r5
        dNdv[1] = 2*r1 - r2
        dNdv[2] = r2
        dNdv[3] = r2 - r3 + r4 - r5
        dNdv[4] = r3  - r4 - r5
        dNdv[5] = r3
        dNdv[6] = r4
        dNdv[7] = 0
        return dNdv
        
    def QSSA(self, v, x):
        """
        List of components
        0 - C2H6, 
        1 - CH4,
        2 - C2H4,
        3 - H2,
        4 - H2O
        """
        
        N1, N3, N6, N7, N8 = x
        Ntot = sum(x)
        Ctot = self.P / (self.R1 * self.T)
        C1 = (N1 / Ntot) * Ctot
        C2 = (2 * self.k[0]) / self.k[1]
        C4 = self.kp * C1
        C5 = self.k[0]/(self.k[4] * self.kp)
    
    
        r1 = self.k[0] * C1
        r2 = 2 * self.k[0] * C1
        r3 = self.k[2] * self.kp * C1
        r4 = self.k[3] * self.k[0] / (self.k[4] * self.kp) * C1
        r5 = self.k[0] * C1
    
        dNdv = np.zeros(5)
        
        # dNdv[0] = -r1 - r2 - r4 + r5;
        dNdv[0] = - r2 - r4;
        dNdv[1] = r2;
        dNdv[2] = r3;
        dNdv[3] = r4;
        dNdv[4] = 0;
        return dNdv

    def simple(self, v, x):
        """
        List of components
        0 - C2H6, 
        1 - C2H4,
        2 - H2,
        3 - H2O
        """

        N1, N6, N7, N8 = x
    
        Ntot = sum(x)
        Ctot = self.P / (self.R1 * self.T)
        C1 = (N1/Ntot) * Ctot
        r = self.k[2] * self.kp * C1
    
        dNdv = np.zeros(4)
    
        dNdv[0] = -r;
        dNdv[1] = r;
        dNdv[2] = r;
        dNdv[3] = 0;

        return dNdv

    def reactor(self):
        # Initial concentrations (gmole/cm3)
        Ptot = self.P_eth + self.P_steam
        C1o = (self.P_eth / Ptot) / (self.R1 * self.T)
        C8o = (self.P_steam / Ptot) / (self.R1 * self.T)
        
        N1o = self.Q * C1o
        N8o = self.Q * C8o
        
        # Volume points for evaluation
        v = np.linspace(0, self.V, 1000)

        # Solve ODE
        if self.model=='exact':
            # Initial state
            Initial = np.array([N1o, 0, N1o*1.2, 0, 0, 0, 0, N8o])
            sol = solve_ivp(lambda v, x: self.exact(v, x), 
                                 [v[0], v[-1]], 
                                 Initial,
                                 method='BDF', 
                                 t_eval=v, 
                                 atol=np.sqrt(np.finfo(float).eps), 
                                 rtol=np.sqrt(np.finfo(float).eps))
            self.solution = sol
            self.ethane = sol.y[0, :]
            self.ethylene = sol.y[5, :]
            self.vol = sol.t
            self.conversion = (self.ethane[0] - self.ethane[-1])/self.ethane[0]
            
        if self.model=='QSSA':
            Initial = np.array([N1o, 0, 0, 0, N8o])
            sol = solve_ivp(lambda v, x: self.QSSA(v, x), 
                                 [v[0], v[-1]], 
                                 Initial,
                                 method='BDF', 
                                 t_eval=v, 
                                 atol=np.sqrt(np.finfo(float).eps), 
                                 rtol=np.sqrt(np.finfo(float).eps))
            self.ethane = sol.y[0, :]
            self.ethylene = sol.y[2, :]
            self.vol = sol.t
            self.solution = sol
            self.conversion = (self.ethane[0] - self.ethane[-1])/self.ethane[0]
            
            
        if self.model=='simple':
            Initial = np.array([N1o, 0, 0, N8o])
            sol = solve_ivp(lambda v, x: self.simple(v, x), 
                                 [v[0], v[-1]], 
                                 Initial,
                                 method='BDF', 
                                 t_eval=v, 
                                 atol=np.sqrt(np.finfo(float).eps), 
                                 rtol=np.sqrt(np.finfo(float).eps))
            self.ethane = sol.y[0, :]
            self.ethylene = sol.y[1, :]
            self.vol = sol.t
            self.solution = sol
            self.conversion = (self.ethane[0] - self.ethane[-1])/self.ethane[0]
    

    def make_plots(self):
        plt.figure(figsize=(10,6))
        plt.title('Ethane Pyrolysis; {} model'.format(self.model))
        plt.plot(self.vol, self.ethane, label='Ethane')
        plt.plot(self.vol, self.ethylene, label='Ethylene')
        plt.ylabel('Molar flow rate (mol/s)')
        plt.xlabel('volume ($cm^3$)')
        plt.legend()
        plt.show()
        
T = 925
V = 200
Q = 35
P = 1
R = 8.314

PFR_exact = PFR_ethane(V, model='exact', Q=Q)
PFR_exact.rate_constants()
PFR_exact.reactor()
# PFR_exact.make_plots()

PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q)
PFR_QSSA.rate_constants()
PFR_QSSA.reactor()
# PFR_QSSA.make_plots()

PFR_simple = PFR_ethane(V, model='simple', Q=Q)
PFR_simple.rate_constants()
PFR_simple.reactor()
# PFR_simple.make_plots()

plt.figure(figsize=(10,7))
plt.title('Ethane pyrolysis comparison')
plt.plot(PFR_QSSA.vol, PFR_QSSA.ethane, '--', color='r', label='QSSA ethane')
plt.plot(PFR_exact.vol, PFR_exact.ethane, color='r', label='Exact ethane')
plt.plot(PFR_QSSA.vol, PFR_QSSA.ethylene, '--', color='b', label='QSSA ethylene')
plt.plot(PFR_exact.vol, PFR_exact.ethylene, color='b', label='Exact ethylene')
plt.plot(PFR_simple.vol, PFR_QSSA.ethane, '--', color='r', label='Simple ethane')
plt.plot(PFR_simple.vol, PFR_QSSA.ethylene, '--', color='b', label='Simple ethylene')

plt.ylabel('Molar flow rate (mol/s)')
plt.xlabel('volume ($cm^3$)')
plt.xlim(0,V)
plt.legend()


plt.figure(figsize=(10,7))  
plt.title('Ethylene QSSA error')
plt.semilogy(PFR_exact.vol[1:], abs(PFR_exact.ethylene[1:] - PFR_QSSA.ethylene[1:])/PFR_exact.ethylene[1:], color='r')
plt.ylabel('QSSA error')
plt.ylabel(r'$ \left| \frac{C_{C_2H_4} - C_{C_2H_4}^*}{C_{C_2H_4}} \right|$')
plt.xlabel('volume ($cm^3$)')
plt.xlim(0,V)
plt.show()

# C0 = (50/760)*(P/R*T)

# exactC0 = np.zeros(8)
# exactC0[0] = C0
# exactRate = abs(PFR_exact.exact(1, exactC0))[0]

# QSSAC0 = np.zeros(5)
# QSSAC0[0] = C0
# QSSARate = abs(PFR_QSSA.QSSA(1, QSSAC0))[0]

# simpleC0 = np.zeros(4)
# simpleC0[0] = C0
# simpleRate = abs(PFR_simple.simple(1, simpleC0))[0]


# exactDa = exactRate*(V/Q)
# QSSADa = QSSARate*(V/Q)
# simpleDa = simpleRate*(V/Q)

#%% Volume variation

v_array = np.linspace(0.01, 100, 100)
v_array = v_array[1:]
exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35
P = 1

for V in v_array:
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P)
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P)
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P)
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)

plt.figure(figsize=(10,7))
plt.title('Volume variation')
plt.plot(v_array, exact_X, color='r', label='Exact model')
plt.plot(v_array, QSSA_X, color='b', label='QSSA model')
plt.plot(v_array, simple_X, color='k', label='simple model')

plt.ylabel(r'Conversion of $C_2H_6$')
plt.xlabel('volume ($cm^3$)')
plt.legend()


#%% Feed variation

v_array = np.linspace(10, 500, 500)
exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35
P = 1

for f in v_array:
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P, feed=[f, 760-f])
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P, feed=[f, 760-f])
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P, feed=[f, 760-f])
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)

plt.figure(figsize=(10,7))
plt.title('Flow-rate variation')
plt.plot(v_array/760, exact_X, color='r', label='Exact model')
plt.plot(v_array/760, QSSA_X, color='b', label='QSSA model')
plt.plot(v_array/760, simple_X, color='k', label='simple model')

plt.ylabel(r'Conversion of $C_2H_6$')
plt.xlabel('Ethane Feed partial pressure')
plt.legend()




#%% Temperature variation

v_array = np.linspace(750, 1000, 100)
v_array = v_array[1:]
exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35

for T in v_array:
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, T=T)
    PFR_exact.rate_constants(T)
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, T=T)
    PFR_QSSA.rate_constants(T)
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, T=T)
    PFR_simple.rate_constants(T)
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)

plt.figure(figsize=(10,7))
plt.title('Temperature variation')
plt.plot(v_array, exact_X, color='r', label='Exact model')
plt.plot(v_array, QSSA_X, color='b', label='QSSA model')
plt.plot(v_array, simple_X, color='k', label='simple model')

plt.ylabel(r'Conversion of $C_2H_6$')
# plt.xlabel('volume ($cm^3$)')
plt.xlabel('Temperature (K)')
plt.legend()


#%% Pressure variation

v_array = np.linspace(0.1, 2, 100)
v_array = v_array[1:]
exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35

for P in v_array:
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P)
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P)
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P)
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)

plt.figure(figsize=(10,7))
plt.title('Pressure variation')
plt.plot(v_array, exact_X, color='r', label='Exact model')
plt.plot(v_array, QSSA_X, color='b', label='QSSA model')
plt.plot(v_array, simple_X, color='k', label='simple model')

plt.ylabel(r'Conversion of $C_2H_6$')
# plt.xlabel('volume ($cm^3$)')
plt.xlabel('Pressure (atm)')
plt.legend()



#%% T and P

p = np.linspace(0.1, 2, 100)
t = np.linspace(750, 1000, 100)

dec_vars = []
for i in p:
    for j in t:
        dec_vars.append(np.array([i,j]))

exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35

for var in dec_vars:
    P, T = var
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P, T=T)
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P, T=T)
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P, T=T)
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)
#%%
ps, ts = np.array(dec_vars).T


plt.figure(figsize=(10,7))
plt.title('Conversion vs T,P')
plt.scatter(ps, ts, c=np.array(exact_X))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')

plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; QSSA scheme')
plt.scatter(ps, ts, c=abs(np.array(exact_X)-np.array(QSSA_X)))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')

ps, ts = np.array(dec_vars).T
plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; Simple scheme')
plt.scatter(ps, ts, c=abs(np.array(exact_X)-np.array(simple_X)))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')



#%%
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

ax = plt.figure().add_subplot(projection='3d')

# Plot the 3D surface
ax.scatter(ps, ts, exact_X, c = exact_X)
ax.scatter(ps, ts, QSSA_X, c = QSSA_X)
ax.scatter(ps, ts, simple_X, c = simple_X)
ax.set_xlabel('Pressure (atm)')
ax.set_ylabel('Temperature (K)')
ax.set_zlabel('Conversion (K)')

#%%

p = np.linspace(0.1, 2, 100)
t = np.linspace(750, 1000, 100)

dec_vars = []
for i in p:
    for j in t:
        dec_vars.append(np.array([i,j]))

exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35

for var in dec_vars:
    P, T = var
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P, T=T)
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P, T=T)
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P, T=T)
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)
#%%
ps, ts = np.array(dec_vars).T


plt.figure(figsize=(10,7))
plt.title('Conversion vs T,P')
plt.scatter(ps, ts, c=np.array(exact_X))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')

plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; QSSA scheme')
plt.scatter(ps, ts, c=abs(np.array(exact_X)-np.array(QSSA_X)))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')

ps, ts = np.array(dec_vars).T
plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; Simple scheme')
plt.scatter(ps, ts, c=abs(np.array(exact_X)-np.array(simple_X)))
plt.colorbar()
plt.ylabel('Temperature (K)')
plt.xlabel('Pressure (atm)')



#%%
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')

# Plot the 3D surface
ax.scatter(ps, ts, exact_X, c = exact_X)
ax.scatter(ps, ts, QSSA_X, c = QSSA_X)
ax.scatter(ps, ts, simple_X, c = simple_X)
ax.set_xlabel('Pressure (atm)')
ax.set_ylabel('Temperature (K)')
ax.set_zlabel('Conversion (K)')
ax.set_title('Conversion vs T,P')





#%%


v = np.linspace(10, 500, 100)
q = np.linspace(10, 500, 100)

dec_vars = []
for i in v:
    for j in q:
        dec_vars.append(np.array([i,j]))

exact_X = []
QSSA_X = []
simple_X = []
V = 100
Q = 35
P = 1
T = 925
for var in dec_vars:
    V, Q = var
    PFR_exact = PFR_ethane(V, model='exact', Q=Q, P=P, T=T)
    PFR_exact.rate_constants()
    PFR_exact.reactor()
    exact_X.append(PFR_exact.conversion)

    PFR_QSSA = PFR_ethane(V, model='QSSA', Q=Q, P=P, T=T)
    PFR_QSSA.rate_constants()
    PFR_QSSA.reactor()
    QSSA_X.append(PFR_QSSA.conversion)

    PFR_simple = PFR_ethane(V, model='simple', Q=Q, P=P, T=T)
    PFR_simple.rate_constants()
    PFR_simple.reactor()
    simple_X.append(PFR_simple.conversion)
#%%
vs, qs = np.array(dec_vars).T


plt.figure(figsize=(10,7))
plt.title('Conversion vs V,Q')
plt.scatter(vs, qs, c=np.array(exact_X))
plt.colorbar()
plt.ylabel('Reactor volume ($cm^3$)')
plt.xlabel('Volumetric flow rate ($cm^3/s$')

plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; QSSA scheme')
plt.scatter(vs, qs, c=abs(np.array(exact_X)-np.array(QSSA_X)))
plt.colorbar()
plt.ylabel('Reactor volume ($cm^3$)')
plt.xlabel('Volumetric flow rate ($cm^3/s$')

ps, ts = np.array(dec_vars).T
plt.figure(figsize=(10,7))
plt.title('Absolute Difference from exact model; Simple scheme')
plt.scatter(vs, qs, c=abs(np.array(exact_X)-np.array(simple_X)))
plt.colorbar()
plt.ylabel('Reactor volume ($cm^3$)')
plt.xlabel('Volumetric flow rate ($cm^3/s$)')



#%%
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d

ax = plt.figure(figsize=(10,8)).add_subplot(projection='3d')

# Plot the 3D surface
ax.scatter(vs, qs, QSSA_X, c = QSSA_X)
ax.scatter(vs, qs, exact_X, c = exact_X)
# ax.scatter(vs, qs, simple_X, c = simple_X)
ax.set_xlabel('Volumetric flow rate ($cm^3/s$)')
ax.set_ylabel('Reactor volume ($cm^3$)')
ax.set_zlabel('Conversion')
ax.set_title('Conversion vs V,Q')
ax.view_init(30, 30)













