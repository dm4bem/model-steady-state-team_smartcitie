print("Smart Cities - final project")

import numpy as np
import dm4bem 
import pandas as pd
import matplotlib.pyplot as plt



# input data setting __________________________________________________________
# set dimensions of house surface areas
l = 1                                                                           # [m]
w = 0.2                                                                         # wall width [m]
h = 2.5                                                                         # height of walls [m]
window = l                                                                      # window length, goes from bottom to top atm
door = 0.9                                                                      # door length. goes from bottom to top atm

# S_ow outdoor wall surface area, total: 3 outdoor walls
S_ow = np.array([4*l, 6*l - window, 11*l + w - window]) * h                     # room order: 1, 2, 3

# S_in indoor wall surface area, total: 5 indoor walls
S_iw = np.array([2*l - door, l - door, 2*l -door, 2*l, 3*l]) * h                # indoor wall order: 0-1, 0-2, 0-3, 1-2, 2-3

# thermo-physical propertites
h_in = 8                                                                        # indoor convection coefficients [W/(m2*K)]
h_out = 25                                                                      # outdoor convection coefficients [W/(m2*K)]

# short-wave solar radiation absorbed by each wall
E = 200                                                                         # [W/m2]

# outdoor temperature
Tout = 0                                                                          # [°C]

# set room temperature (same setpoint for all rooms)
Tin=22                                                                # [°C]                                                                   # [°C]

# set controller gain
# ADD TO REPORT:    small controller gain --> takes long to reach final temperature or may not reach
#                   high controller gain means very low temperature differnce and you reach final temperature quickly and precise
# BUT, high controller gain means in reality to oversize the system: whats the best trade off? find smallest controller gain to reach indoor temperatures  
contr_g1 = 200                                                                  # set controller gain 1
contr_g2 = 200                                                                  # set controller gain 2
contr_g3 = 200                                                                  # set controller gain 3

# Advection/Ventilation
ACH = 1                                                                         # ventilation rate [volumes/hour] // different ventilation rates at course website
ρ_air = 1.2                                                                     # density air in [kg/m3]
c_air = 1000                                                                    # specific heat capacity of air in [J/kg*K]

# Wall properties
λ_concrete = 1.4                                                                # thermal conductivity [W/(m*K)]
ρ_concrete = 2300                                                               # density in [kg/m3]
c_concrete = 880                                                                # specific heat in [J/(kg⋅K)]

λ_insulation = 0.027                                                            # thermal conductivity [W/(m*K)]
ρ_insulation = 55                                                               # density in [kg/m3]
c_insulation = 1280                                                             # specific heat in [J/(kg⋅K)]

λ_glass = 1.4                                                                   # wall thermal conductivity [W/(m*K)]
ρ_glass = 2500                                                                  # density in [kg/m3]
c_glass = 1210                                                                  # specific heat in [J/(kg⋅K)]


# thermal mode
#V_dot = np.array([l*2*l, 4*l*l, 9*l*l, 3*l*(5*l+w)]) * h * ACH / 3600          # volumetric air flow rate [m3/s], room 0, 1, 2 and 3
m_dot = np.array([l*2*l, 4*l*l, 9*l*l, 3*l*(5*l+w)]) * h * ACH / 3600 * ρ_air       # mass air flow rate [kg/s] // m_dot = V_dot * ρ_air

# set number of nodes and flow rates
n_branches = 32
n_nodes = 17


# incidence matrix A __________________________________________________________
A = np.zeros([n_branches, n_nodes])

# outdoor convection, branches 0 to 2
A[0,6] = 1
A[1,9] = 1
A[2,16] = 1

# thermal bridges, branches 3 to 6
A[3,1] = 1
A[4,2] = 1
A[5,3] = 1
A[6,0] = 1

# controllers, branches 7 to 9
A[7,1] = 1
A[8,2] = 1
A[9,3] = 1

# outdoor conduction (and convection), branches 10 to 16
A[10,5] = 1; A[10,6] = -1 
A[11,1] = 1; A[11,5] = -1
A[12,8] = 1; A[12,9] = -1
A[13,2] = 1; A[13,8] = -1
A[14,15] = 1; A[14,16] = -1
A[15,14] = 1; A[15,15] = -1
A[16,3] = 1; A[16,14] = -1

# indoor convection, conduction, convection, branches 17 to 27
A[17,4] = 1; A[17,1] = -1
A[18,0] = 1; A[18,4] = -1
A[19,7] = 1; A[19,1] = -1
A[20,2] = 1; A[20,7] = -1
A[21,10] = 1; A[21,2] = -1
A[22,0] = 1; A[22,10] = -1
A[23,11] = 1; A[23,2] = -1
A[24,12] = 1; A[24,11] = -1
A[25,3] = 1; A[25,12] = -1
A[26,13] = 1; A[26,0] = -1
A[27,3] = 1; A[27,13] = -1

# ventilation, branches 28 to 31
A[28,1] = 1
A[29,0] = 1; A[29,1] = -1
A[30,3] = 1; A[30,0] = -1
A[31,3] = 1


# conductance matrix G _________________________________________________________
G=np.zeros([n_branches])

# outdoor convection, branches 0 to 2
G[0:3] = h_out * S_ow                                                   

# thermal bridge conduction
G[3:7] = 0.5 * h

# controllers, branches 7 to 9
# no controller used: 0 // "free running"
# controller used: 1e9 (for example)
# controllers used: G_contr = Kp (proportional gain)
G[7] = contr_g1                                                                 # controller 1 gain
G[8] = contr_g2                                                                 # controller 2 gain
G[9] = contr_g3                                                                 # controller 3 gain

# outdoor walls, conduction or conduction + convection, branches 10 to 16
G[10] = λ_concrete / (w/2) * S_ow[0]                                            # conduction of wall (half), room 1
G[11] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_ow[0]                                  # conduction + indoor convection (half wall), room 1
G[12] = λ_concrete / (w/2) * S_ow[1]                                                     # conduction of wall (half), room 2
G[13] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_ow[1]                                  # conduction + indoor convection (half wall), room 2
G[14] = λ_concrete / (w/2) * S_ow[2]                                                     # conduction of wall (half), room 3
G[15] = λ_concrete / (w/2) * S_ow[2]                                                     # conduction of wall (half), room 3
G[16] = h_in * S_ow[2]                                                          # inside convection of outdoor wall, room 3

# indoor walls, convection + conduction, branches 17 to 27
G[17] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[0]                                  # conduction + convection, half indoor wall, room 0-1
G[18] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[0]                                  # conduction + convection, half indoor wall, room 0-1
G[19] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[3]                                  # conduction + convection, half indoor wall, room 1-2
G[20] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[3]                                  # conduction + convection, half indoor wall, room 1-2
G[21] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[1]                                  # conduction + convection, half indoor wall, room 0-2
G[22] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[1]                                  # conduction + convection, half indoor wall, room 0-2
G[23] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[4]                                  # conduction + convection, half indoor wall, room 2-3
G[24] = λ_concrete / (w/2) * S_iw[4]                                                     # conduction, half indoor wall, room 2-3
G[25] = h_in * S_iw[4]                                                          # convection, half indoor wall, room 2-3
G[26] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[2]                                  # conduction + convection, half indoor wall, room 0-3
G[27] = (1 / ((w/2) / λ_concrete + 1 / h_in)) * S_iw[2]                                  # conduction + convection, half indoor wall, room 0-3


# Advection due to Ventilation
# no ventilation: 0
# ventilation: m_dot * c
# in our case - ventilation flow: outdoor -> room 3 -> room 0 -> room 1 -> leaves room (ventilation system)
# conductances G31, G30, G29 = m_dot*c; G28 = 0
G[28] = 1
G[29] = m_dot[3] * c_air                                                            # not sure, if we use the volume of room 3 or another one at m_dot
G[30] = m_dot[3] * c_air
G[31] = m_dot[3] * c_air

# Capacity matrix C _________________________________________________
C=np.zeros([n_nodes])

#(unlike for steady state, we now consider the thermal inertia of the building, 
#which means that we must calculate the thermal capacities of the walls, air in the room,
#glass doors)

# C = ρ (air or wall material) * c_wall * w * S
#Room air capacities
C[0] = ρ_air * c_air * 3*l*l*h
C[1] = ρ_air * c_air * 4*l*l*h
C[2] = ρ_air * c_air * 9*l*l*h
C[3] = ρ_air * c_air * 3*l*(5*l+3*w)*h

# Wall concrete capcities
# no windows considered, doors are made of glass
C[4] = ρ_concrete * c_concrete * l*w*h + ρ_glass * c_glass * l*0.01*h           # wall and door
C[5] = ρ_concrete * c_concrete * (4*l+w)*w*h
C[7] = ρ_concrete * c_concrete * 2*l*w*h
C[8] = ρ_concrete * c_concrete * (8*l+w)*w*h + ρ_glass * c_glass * l*0.01*h     # wall and window
C[10] = ρ_glass * c_glass * l*w*h                                               # glass door
C[11] = ρ_concrete * c_concrete * 3*l*w*h
C[13] = ρ_concrete * c_concrete * l*w*h + ρ_glass * c_glass * l*w*h             # wall and door
C[15] = ρ_concrete * c_concrete * (10*l+5*w)*w*h + ρ_glass * c_glass * l*w*h    # wall and window


#settling time and the step time is affected by the capcities:
    # increasing the capacities of the walls e.g. concrete will increase the settling time
    #but it will not significantly affect the step time

#define q labels and theta labels for use in the state-space representation input dataframes
q_l = [f'q{i}' for i in range(n_branches)]
θ_l = [f'θ{j}' for j in range(n_nodes)]


#vectors b and f: unlike before this is not a vector of value but instead a series of strings that are later used to the variables
# vector b
b = pd.Series(['To', 'To', 'To', 0,0,0,0,'Ti_sp','Ti_sp','Ti_sp',0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,'To',0,0,'To'],index=q_l)
# heatflow rate sources vector f _________________
f = pd.Series([0, 0, 0, 0, 0, 0, 'Φo', 0, 0, 'Φo', 0, 0, 'Φi', 0, 'Φi', 0, 'Φo'], index=θ_l)

#output vector (same as for steady-state)
y = np.zeros([n_nodes])         # nodes
y[[0,1,2,3]] = 1               # nodes (temperatures) of interest
#pd.DataFrame(y,index=θ_l)

#TC as dictionary for the thermal network:
 
# thermal circuit conversion to pandas dataframes to then enter into the TC structure 
#used for the dm4bem tc2ss function
A = pd.DataFrame(A, index=q_l, columns=θ_l)
G = pd.Series(G, index=q_l)
C = pd.Series(C, index=θ_l)
b = pd.Series(b, index=q_l)
f = pd.Series(f, index=θ_l)
y = pd.Series(y, index=θ_l)


TC = {"A": A,
      "G": G,
      "C": C,
      "b": b,
      "f": f,
      "y": y}

# State-space representation
[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

#Time step response
egv = np.linalg.eig(As)[0]
print('eigenvalues:\n', egv)

#time step calculation:
dt = 2* min(-1. / egv)
dm4bem.print_rounded_time('dt_max', dt)


#settling time:
t_set= 4 * max(-1 / egv)
dm4bem.print_rounded_time('t_set', t_set)


#simulation duration
t_sim=np.ceil(t_set/3600)*3600
dm4bem.print_rounded_time('duration t_sim', t_sim)

#Input vector for time series
n=int(np.floor(t_sim/dt)  )          #defining the number of time steps

#OVERIDE timestep for shorter simulation time
#n=10000
    
#make a date time index vector with the delta t timestep
time=pd.date_range(start="2024-01-01 00:00:00",
                   periods = n, freq=f"{int(dt)} S")
     
# absorbed short-wave solar radiation [W/m2] * wall surface [m2]
walls_rad_out = E * (3*l + 3*w + 5*l)

# absorbed short-wave solar radiation [W/m2] * wall surface [m2]
#walls_rad_out_in = E * 3*l*h                                                 #S_ow[2] // instead of the whole inside wall of room 3, only the top wall (3*l)
walls_rad_in = E * S_iw[4]


#outdoor temp
To=Tout*np.ones(n)          #outdoor temperature vector- contains the n entries
Ti_sp=Tin*np.ones(n)        #indoor temp vector
Φa = 0 * np.ones(n)         #solar radiation on indoor surface of outdoor walls
Φi= walls_rad_in * np.ones(n)       #indoor walls solar radiation
Φo= walls_rad_out * np.ones(n)   #outside walls (all) solar radiation
Qa=Φa                           #auxiliary heat sources - in our case we have no auxiliary sources so set
                                    #set this vector to 0 and the same for phi a
data = {'To': To, 'Ti_sp': Ti_sp, 'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa}
input_data_set = pd.DataFrame(data, index=time)

# inputs in time from input_data_set
u = dm4bem.inputs_in_time(us, input_data_set)


# Initial conditions simulation
θ_exp = pd.DataFrame(index=u.index)     # empty df with index for explicit Euler
θ_imp = pd.DataFrame(index=u.index)     # empty df with index for implicit Euler

θ0 = 0.0                    # initial temperatures
θ_exp[As.columns] = θ0      # fill θ for Euler explicit with initial values θ0
θ_imp[As.columns] = θ0      # fill θ for Euler implicit with initial values θ0

I = np.eye(As.shape[0])     # identity matrix
for k in range(u.shape[0] - 1):
    θ_exp.iloc[k + 1] = (I + dt * As)\
        @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k]
    θ_imp.iloc[k + 1] = np.linalg.inv(I - dt * As)\
        @ (θ_imp.iloc[k] + dt * Bs @ u.iloc[k])

# outputs
y_exp = (Cs @ θ_exp.T + Ds @  u.T).T
y_imp = (Cs @ θ_imp.T + Ds @  u.T).T

# plot results
y = pd.concat([y_exp, y_imp], axis=1, keys=['Explicit', 'Implicit'])
# Flatten the two-level column labels into a single level
y.columns = y.columns.get_level_values(0)

ax = y.plot()
ax.set_xlabel('Time')
ax.set_ylabel('Indoor temperature, $\\theta_i$ / °C')
ax.set_title(f'Time step: $dt$ = {dt:.0f} s')
plt.show()



print('End of program reached')