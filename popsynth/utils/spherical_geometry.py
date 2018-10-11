import numpy as np

def sample_theta_phi(size):
    
    theta = np.arccos(1- 2*np.random.uniform(0.,1., size=size) )
    phi = np.random.uniform(0, 2*np.pi, size=size)
    
    return theta, phi


def xyz(r,theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    
    return x,y,z
