#kalman.py
#July 6th, 2016

import numpy as np
import matplotlib.pyplot as plt

#simulation variables
dt = .1
tot_t = 7
timesteps = tot_t/dt
msmt_rate = .5

#covariance matricies 
R = np.array([[.3, 0], [0, 1]]) #measuement Covariance
Q = np.array([[3, 0, 0], [0, .1, 0],[0, 0, .01]]) #process covariance

#initial conditions
x = np.array([0, 30, -9.8])
p = np.eye(3)
z = np.zeros((3,1))

#Transition Matricies
A = np.array([[1, dt, dt*dt/2], [0, 1, dt], [0, 0, 1]]) #state transition
H = np.array([[1, 0, 0], [0, 1, 0]]) #measurement "transistion"

#Kalman Gain Matrix
K = np.zeros((3,2))

#Matricies for plotting later
X = np.zeros((3,0))
Z = np.zeros((2,0))
T = []

time = np.linspace(dt, tot_t, timesteps)

#Calculate the actual state of the object under the force of gravity, with covariance matrix Q
act_state = np.array([0+ 30*time -.5*time*time*9.8, 30 - 9.8*time, -9.8*np.ones(tot_t/dt)]) + np.random.multivariate_normal(np.zeros(3),Q, (70)).T

def take_msmt(state, R):
	return np.dot(H, state) + np.random.multivariate_normal(np.zeros(2),R)

def priori_update_state(state, A):
	return np.dot(A,x)

def priori_update_covar(A,p,Q):
	return np.dot(np.dot(A, p),A.T) + Q

def calc_gain(p,H,R):
	return np.dot(np.dot(p,H.T), np.linalg.inv(np.dot(np.dot(H,p),H.T) + R))

def update_state(x,K,z,H):
	return x + np.dot(K, (z - np.dot(H,x)))

def update_covar(K,H,p):
	return np.dot((np.eye(3) - np.dot(K,H)), p)

for i in range(int(timesteps)):
	current_time = time[i]

	#Time update
	x = priori_update_state(x, A)
	p = priori_update_covar(A, p, Q)

	#measurement update, when data is "ready"
	if (current_time%msmt_rate <= .000001): #floating point numbers occasionally throw off equality checking
		print current_time
		z = take_msmt(act_state.T[i].T, R)
		K = calc_gain(p, H, R)
		x = update_state(x, K, z, H)
		p = update_covar(K,H, p)
		Z = np.concatenate((Z, z.reshape(2,1)), axis = 1)
		T.append(current_time) #attach time to master list of times

	X = np.concatenate((X, x.reshape(3,1)), axis = 1) #attach state vector to master list of states

# Plot position, velocity, acceleration
plt.figure(figsize = (18,6))

#plot position
plt.subplot(1, 3, 1)
plt.plot(time, act_state[0],'r', label = 'Actual')
plt.plot(time, X[0],'g', label = 'Estimated')
plt.plot(T, Z[0],'ko', label = 'Measured')
plt.legend()
plt.xlabel('time (s)')
plt.ylabel('Position')

#plot Velocity
plt.subplot(1, 3, 2)
plt.plot(time, act_state[1],'r', label = 'Actual')
plt.plot(time, X[1],'g', label = 'Estimated')
plt.plot(T, Z[1],'ko', label = 'Measured')
plt.legend()
plt.title('Kalman Filtering')
plt.xlabel('time (s)')
plt.ylabel('Velocity')

#plot Acceleration
plt.subplot(1, 3, 3)
plt.plot(time, act_state[2],'r')
plt.plot(time, X[2],'g')
plt.xlabel('time (s)')
plt.ylabel('Acceleration')

plt.show()



