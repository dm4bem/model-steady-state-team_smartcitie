print("Smart Cities - final project")

import numpy as np


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
To = 0                                                                          # [°C]

# set room temperature
T_r1 = 25                                                                       # [°C]
T_r2 = 25                                                                       # [°C]
T_r3 = 25                                                                       # [°C]

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
A = np.zeros([n_branches, n_nodes])                                             #set all node-branch relations

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

# temperature sources vector b _________________________________________________
b = np.zeros([n_branches])

# outdoor temperatures for outdoor walls 
b[0:3] = To

# outdoor temperature for ventilation (going in and going out)
b[28] = To
b[31] = To

# setpoints temperatures for rooms (room 0 not controlled)
b[7] = T_r1                                                                     # set temperature room 1
b[8] = T_r2                                                                     # set temperature room 2
b[9] = T_r3                                                                     # set temperature room 3

# heatflow rate sources vector f _______________________________________________
f=np.zeros([n_nodes])

# solar radiation @ outside of outdoor walls: θ = 6 (room 1), θ = 9 (room 2), θ = 16 (room 3)
walls_rad_out = [6, 9, 16]

# solar radiation @ inside of outdoor walls: θ = 14 (room 3)
walls_rad_out_in = [14]                                                         # computations: very high temperatures on the inside wall of room 3 because we assume the solar ardiation heats it up but it is not realistic

# solar radiation @ indoor wall: θ = 12 ( wall room 2-3)
walls_rad_in = [12]

# absorbed short-wave solar radiation [W/m2] * wall surface [m2]
f[walls_rad_out] = E * S_ow
# discuss with teacher 
f[walls_rad_out_in] = E * 3*l*h                                                 #S_ow[2] // instead of the whole inside wall of room 3, only the top wall (3*l)
f[walls_rad_in] = E * S_iw[4]


# computation __________________________________________________________________
Gd = np.diag(G)
A_T_Gd_A = A.T @ Gd @ A

# compute tmperatures nodes and heat flows
θ = np.linalg.inv(A_T_Gd_A) @ (A.T @ Gd @ b + f)
q = Gd @ (-A @ θ + b)


# outputs ______________________________________________________________________
temp_rooms = [0, 1, 2, 3]                                                       # room air temperatures
temp_out_walls = [5, 6, 8, 9, 14, 15, 16]                                       # outdoor walls temperatures
temp_ind_walls = [4, 7, 10, 11, 12, 13]                                         # indoor walls temperatures
heat_flow_controller = [7, 8, 9]                                                # controller branches
heat_flow_out_walls = [10, 11, 12, 13, 14, 15, 16]                              # outdoor wall branches
heat_flow_ind_walls = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]              # indoor wall branches
heat_flow_out_conv = [0, 1, 2]

# output printing
print()
print("Computed temperatures")
print(f"Air temp. of rooms  θ[0, 1, 2, 3]:              {[f'{temp:.2f}' for temp in θ[temp_rooms]]} °C")
print(f"Outdoor wall temp.  θ[5, 6, 8, 9, 14, 15, 16]:  {[f'{temp:.2f}' for temp in θ[temp_out_walls]]} °C")
print(f"Indoor wall temp.   θ[4, 7, 10, 11, 12, 13]:    {[f'{temp:.2f}' for temp in θ[temp_ind_walls]]} °C")
print()
print("Computed heat flows")
print(f"heat flow controllers   q[7, 8, 9]:                                     {[f'{flow:.2f}' for flow in q[heat_flow_controller]]} W")
print(f"heat flow outdoor walls q[10, 11, 12, 13, 14, 15, 16]:                  {[f'{flow:.2f}' for flow in q[heat_flow_out_walls]]} W")
print(f"heat flow indoor walls  q[17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]:  {[f'{flow:.2f}' for flow in q[heat_flow_ind_walls]]} W")        # if room temperatures are equal, heat flows should be 0
print(f"heat flow outdoor conv  q[0, 1, 2]:                                     {[f'{flow:.2f}' for flow in q[heat_flow_out_conv]]} W")
print()