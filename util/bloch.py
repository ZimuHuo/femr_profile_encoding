'''
Simualtion code 
Author: Zimu Huo
Date: 02.2022
'''
import sys
sys.path.insert(1, '../')
sys.path.insert(1, '../../')
import numpy as np
from util.fft import *
from fractions import Fraction
def iterative_SSFP(M0, alpha, phi, dphi, beta, TR, TE, T1, T2):
    signal = np.zeros([1,3])
    signal = np.asarray([0, 0, M0])
    Nr = len(TR)
    for i in range(Nr-1):
        signal = np.matmul(rotMat(alpha[i], phi[i]+np.sum(dphi[0:i])),signal) 
        signal[0] = signal[0]*np.exp(-TR[i]/T2)
        signal[1] = signal[1]*np.exp(-TR[i]/T2)
        signal[2] = M0+(signal[2]-M0)*np.exp(-TR[i]/T1)
        P = np.array([
            [ np.cos(beta[i]),  np.sin(beta[i]),   0],
            [-np.sin(beta[i]),  np.cos(beta[i]),   0],
            [            0,             0,   1]
            ])
        signal = np.matmul(P,signal)
        
    signal = np.matmul(rotMat(alpha[-1], phi[-1]+np.sum(dphi[:-1])),signal) 
    
    signal[0] = signal[0]*np.exp(-TE/T2)
    signal[1] = signal[1]*np.exp(-TE/T2)
    signal[2] = M0+(signal[2]-M0)*np.exp(-TE/T1) 
    P = np.array([
            [ np.cos(beta[-1]),  np.sin(beta[-1]),   0],
            [-np.sin(beta[-1]),  np.cos(beta[-1]),   0],
            [            0,             0,      1]
            ])
    signal = np.matmul(P,signal)
    return signal
def bssfp_bloch(b, size,  M0, alpha, phi, dphi, TR, TE, T1, T2):
    samples = np.zeros([size,3])
    Nr = len(TR)
    for index, beta in enumerate((b)):
        # some safty checks 
        betas = np.ones(Nr) * beta * TR[:Nr] / np.max(TR[:Nr])
        betas[-1] = beta * TE / np.max(TR[:Nr])
        ntime = int(Nr)
        mx = np.zeros(Nr)
        my = np.zeros(Nr)
        mz = np.zeros(Nr)
        alpha = np.asarray(alpha).astype("float64")[:Nr]
        beta = np.asarray(betas).astype("float64")[:Nr]
        TR = np.asarray(TR).astype("float64")[:Nr]
        phi = np.asarray(phi).astype("float64")[:Nr]
        dphi = np.asarray(dphi).astype("float64")[:Nr]
        
        samples[index,:] = iterative_SSFP(M0 = M0 , alpha = alpha, phi = phi, dphi = dphi, beta = betas, TR= TR, TE= TE, T1 = T1, T2 = T2)
    data = np.zeros([size,1], dtype = complex)
    data.real = samples[:,1].reshape(-1,1)
    data.imag = samples[:,0].reshape(-1,1)
    axis = b / (2*np.pi*TR[0]/1000)
    return axis, data


def plot_bssfp(x, data):
    magnitude, phase = np.absolute(data), np.angle(data)
    plt.figure()
    plt.subplot(211)
    plt.ylabel('Magitude')
    plt.title('Sequence')
    plt.grid(True)
    plt.plot(x, magnitude)
    plt.subplot(212)
    plt.xlabel('Off-Resonance (Hz)')
    plt.ylabel('Phase')
    plt.grid(True)
    plt.plot(x, phase)
    plt.show()
def rf_spoiler(a):
    Nr = np.pi / a * 2 
    dphi = 0.5 * a * np.arange(Nr)**2
    return dphi.tolist()    
def bssfp_trace(M0, alpha, phi, dphi, beta, TR, TE, T1, T2):
    output = []
    signal = np.zeros([1,3])
    signal = np.asarray([0, 0, M0])
    output.append(signal)
    Nr = len(TR)
    for i in range(Nr-1):
        signal = np.matmul(rotMat(alpha[i], phi[i]+np.sum(dphi[0:i])),signal) 
        signal[0] = signal[0]*np.exp(-TR[i]/2/T2)
        signal[1] = signal[1]*np.exp(-TR[i]/2/T2)
        signal[2] = M0+(signal[2]-M0)*np.exp(-TR[i]/2/T1)
        P = np.array([
            [ np.cos(beta[i]/2),  np.sin(beta[i]/2),   0],
            [-np.sin(beta[i]/2),  np.cos(beta[i]/2),   0],
            [            0,             0,   1]
            ])
        signal = np.matmul(P,signal)
        output.append(signal)
        signal[0] = signal[0]*np.exp(-TR[i]/2/T2)
        signal[1] = signal[1]*np.exp(-TR[i]/2/T2)
        signal[2] = M0+(signal[2]-M0)*np.exp(-TR[i]/2/T1)
        signal = np.matmul(P,signal)
    signal = np.matmul(rotMat(alpha[-1], phi[-1]+np.sum(dphi[:-1])),signal) 
    
    signal[0] = signal[0]*np.exp(-TE/T2)
    signal[1] = signal[1]*np.exp(-TE/T2)
    signal[2] = M0+(signal[2]-M0)*np.exp(-TE/T1) 
    P = np.array([
            [ np.cos(beta[-1]),  np.sin(beta[-1]),   0],
            [-np.sin(beta[-1]),  np.cos(beta[-1]),   0],
            [            0,             0,      1]
            ])
    signal = np.matmul(P,signal)
    output.append(signal)
    return np.asarray(output)[:,1],  np.asarray(output)[:,0], np.asarray(output)[:,2]

def rotMat(alpha, phi):
    rotation = np.array([
    [np.cos(alpha)*np.sin(phi)**2 + np.cos(phi)**2,          
         (1-np.cos(alpha))*np.cos(phi)*np.sin(phi),         
     -np.sin(alpha)*np.sin(phi)],
        
    [    (1-np.cos(alpha))*np.cos(phi)*np.sin(phi),        
       np.cos(alpha)*np.cos(phi)**2+np.sin(phi)**2,         
                        np.sin(alpha)*np.cos(phi)],
        
    [                    np.sin(alpha)*np.sin(phi),                      
                        -np.sin(alpha)*np.cos(phi),                    
                                     np.cos(alpha)]
    ])
    return rotation

# def vectorform_SSFP(M0, alpha, phi, dphi, beta, TR, TE, T1, T2, Nr):
#     #M = np.asarray([0, 0, M0])
    
#     '''
#     -------------------------------------------------------------------------
#     Parameters
    
#     M0: array_like  
#     Initial magnetization in the B0 field, function of the proton density rho represented by [x, y, z]  
    
#     Alpha: radian 
#     Flip or tip angle of the magnetization vector 
    
#     phi: radian 
#     Angle between the vector and x axis/ phase
    
#     dphi: radian 
#     Increment of phi
    
#     TR: scalar in msec  
#     Repition time of the pulse sequence 
    
#     TE: scalar in msec  
#     Echo time 
    
#     T1: scalar in msec
#     T1 value of the tissue
    
#     T2: scalar in msec
#     T2 value of the tissue
    
#     Nr: scalar 
#     Number of simulation   
    
#     -------------------------------------------------------------------------
#     Returns
#     Signal : single complex value
#     x component as real and y component as imag
    
#     -------------------------------------------------------------------------
#     Notes
#     Neal's thesis chapter 3.2 equation 3.9
#     The key is that in steady state, the M+ = M-, which allows analytical solution 
    
#     -------------------------------------------------------------------------
#     References
    
#     [1] 
#     Author: Dr Neal K Bangerter
#     Title: Contrast enhancement and artifact reduction in steady state magnetic resonance imaging
#     Link: https://www.proquest.com/openview/41a8dcfb0f16a1289210b3bd4f9ea82b/1.pdf?cbl=18750&diss=y&pq-origsite=gscholar
#     '''

#     assert alpha != 0, 'Only non zero numbers are allowed'
#     assert T1 != 0, 'Only non zero numbers are allowed'
#     assert T2 != 0, 'Only non zero numbers are allowed'
#     assert TE != 0, 'Only non zero numbers are allowed'
#     assert TR != 0, 'Only non zero numbers are allowed'
#     assert TE <= TR, 'TE must be shorter than or equal to TR'
    
#     M = M0
#     I = np.identity(3)
#     E = np.diag([np.exp(-TR/T2), np.exp(-TR/T2), np.exp(-TR/T1)])
#     ETE = np.diag([np.exp(-TE/T2), np.exp(-TE/T2), np.exp(-TE/T1)])
#     P = np.array([[np.cos(beta),np.sin(beta),0], [-np.sin(beta),np.cos(beta),0], [0,0,1]])
#     PTE = np.array([[np.cos(beta*TE/TR), np.sin(beta*TE/TR), 0],
#                    [-np.sin(beta*TE/TR), np.cos(beta*TE/TR), 0],
#                    [                  0,                  0, 1]])
#     R = rotMat(alpha, phi)
#     #(I-P*E*Rx)^-1*(I-E)*M0     
#     #Mneg = np.matmul(np.linalg.inv(I - np.matmul(P,np.matmul(E, Rx))), np.matmul(I-E, M0))
    
#     #equation 3.7
#     Mneg = np.linalg.inv((I-P@E@R))@(I-E)@M            
    
#     #equation 3.8
#     Mpos = R@Mneg                                      
    
#     #equation 3.9
#     MTE = PTE@ETE@Mpos + (I-ETE)@M 
#     data = np.zeros([1], dtype = complex)
#     data.real = MTE[0]
#     data.imag = MTE[1]
#     return data



