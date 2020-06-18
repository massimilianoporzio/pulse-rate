#!/usr/bin/env python
# coding: utf-8

# # Test Your Algorithm
# 
# ## Instructions
# 1. From the **Pulse Rate Algorithm** Notebook you can do one of the following:
#    - Copy over all the **Code** section to the following Code block.
#    - Download as a Python (`.py`) and copy the code to the following Code block.
# 2. In the bottom right, click the <span style="color:blue">Test Run</span> button. 
# 
# ### Didn't Pass
# If your code didn't pass the test, go back to the previous Concept or to your local setup and continue iterating on your algorithm and try to bring your training error down before testing again.
# 
# ### Pass
# If your code passes the test, complete the following! You **must** include a screenshot of your code and the Test being **Passed**. Here is what the starter filler code looks like when the test is run and should be similar. A passed test will include in the notebook a green outline plus a box with **Test passed:** and in the Results bar at the bottom the progress bar will be at 100% plus a checkmark with **All cells passed**.
# ![Example](example.png)
# 
# 1. Take a screenshot of your code passing the test, make sure it is in the format `.png`. If not a `.png` image, you will have to edit the Markdown render the image after Step 3. Here is an example of what the `passed.png` would look like 
# 2. Upload the screenshot to the same folder or directory as this jupyter notebook.
# 3. Rename the screenshot to `passed.png` and it should show up below.
# ![Passed](passed.png)
# 4. Download this jupyter notebook as a `.pdf` file. 
# 5. Continue to Part 2 of the Project. 

# In[3]:


# replace the code below with your pulse rate algorithm.
import numpy as np
import math
import cmath
import glob
import scipy as sp
from scipy import stats
import scipy.io
import scipy.signal
from numpy import linalg as LA

LOW_BPM_CUT = 40/60
HIGH_BPM_CUT = 240/60

LOW_BPM = 40 # min freq for heart rate
HIGH_BPM = 240 #max freq heart rate


def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]

def AggregateErrorMetric(pr_errors, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Computes the MAE at 90% availability. 

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        the MAE at 90% availability
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]
    
    # Return the mean absolute error
    return np.mean(np.abs(best_estimates))

def Evaluate():
    """
    Top-level function evaluation function.

    Runs the pulse rate algorithm on the Troika dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error on the Troika dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    data_fls, ref_fls = LoadTroikaDataset()
    errs, confs = [], []
    for data_fl, ref_fl in zip(data_fls, ref_fls):
        # Run the pulse rate algorithm on each trial in the dataset
        errors, confidence = RunPulseRateAlgorithm(data_fl, ref_fl)
        errs.append(errors)
        confs.append(confidence)
        # Compute aggregate error metric
    errs = np.hstack(errs)
    confs = np.hstack(confs)
    return AggregateErrorMetric(errs, confs)


class Signal():
    """
    A class to represent a time-signal enseble of one ppg and three accelerometer signals (x,y and z)
    
    Attributes
    -------------
    ppg1 : numpy array
            the ppg signal (time-signal)
    x : numpy array
            the accelerometer signal along x-axis 
    y : numpy array
            the accelerometer signal along y-axis 
    z : numpy array
            the accelerometer signal along z-axis
    
    """
    ppg1 = np.array
    ppg2 = np.array
    x = np.array
    y = np.array
    z = np.array
    
    def __init__(self,_ppg1,accx,accy,accz):
        """
        Parameters
        ----------
        
        ppg1 : numpy array
            the ppg signal (time-signal)
        accx : numpy array
            the accelerometer signal along x-axis 
        accy : numpy array
            the accelerometer signal along y-axis 
        accz : numpy array
            the accelerometer signal along z-axis
        """
         
        self.ppg1 =_ppg1
        self.x = accx
        self.y = accy
        self.z = accz
    
    def get_acc_magn_sig(self):
        """Return the magnitude signal of the accelerometer (numpy array)
        
        
        """
        acc_data = np.zeros((3,len(self.x)))
        acc_data[0] = self.x
        acc_data[1] = self.y
        acc_data[2] = self.z
        acc_mag = LA.norm(acc_data,axis=0)
        return acc_mag
    
    def get_Y_matrix(self):
        """Return the Y matrix obtained using ppg,accx,accy,accz
        
        
        """
        result = np.zeros((4,len(self.x)))
        result[0] = self.ppg1
        result[1] = self.x
        result[2] = self.y
        result[3] = self.z
        return result

def FOCUSS(sig,Phi):
    """
    Construct a sparse spectrum of y using the FOCUSS algorithm
    
    '''
    ---------
    Parameters
    sig : numpy array
        time-signal (like ppg and/or accelerometer signals)
    Phi : numpy 2D array 
        Fourier matrix
    
    
    
    ---------
    Output:
    s : numpy array
    """
    #number of iterations
    iterations = 2
    #resolution of the spectrum
    N = Phi.shape[1]
    
    #initialization of x
    x = np.ones((N,1))
    
    for i in range(iterations):
        W_pk = np.diagflat(x)
        q_k = (LA.pinv(Phi @ W_pk)) @ sig
       # q_k = (MPinverse(Phi @ W_pk))@ sig
        x = W_pk @ q_k
        s = (np.absolute(x))**2
    return s

def BP(SSRsig,fs,BPM_bp):
    """ Band pass filter the signal sparse spectrum
   
    '''
    --------
    Parameters
   
    SSRsig : numpy array
        sparse spectrum of the signal
    fs : int
        sampling frequency (Hz)
    BPM_bp : tuple of (int, int)
        Band-pass frequencies in BPM: (lower,upper)
   
    -------
    Output
    numpy array - the filtered spectrum
       
    """
    N = len(SSRsig)
    
    f_lo = BPM_bp[0]/60
    f_up = BPM_bp[1]/60
    idx_lo,idx_up =(math.floor(f_lo/fs*N), math.ceil(f_up/fs*N+1))
    
    H = np.zeros(N)
    H[idx_lo:idx_up] = 1
    result = np.multiply(SSRsig,H)
    
    return result

def SSR(sig,fs,N=None):
    """
    Return the Sparse spectrum (filtered/band-passed) and the frequency axis in BPM
    
    '''
    Parameters
    sig : numpy array
            time-signal (ppg or accelerometer)
    fs :  int
            sampling frequency
    N : int
            resolution (default value (if N = None) = 60*fs in order to obtain a 1 BPM resolution 
    """
    #decimating the signal
   # sig = scipy.signal.decimate(sig,10,zero_phase=True)
    
    M = len(sig)
    if N is None:
        N = (int)( 60*fs)
    #construct Fourier  matrix
    Phi = np.zeros((M,N),dtype=complex)
    complex_factor = 1j*2*np.pi/N;
    for m in range(1,M+1):
        for n in range(1,N+1):
            Phi[m-1,n-1] = cmath.exp(complex_factor*(m-1)*(n-1))
    #calculate BPM-axis
    BPM = 60*fs/N*(np.arange(1,N+1)-1)
    #construct sparse spectrum
    s = FOCUSS(sig,Phi)
    
    #band pass freq between 40 and 240
    s = BP(s,fs,(LOW_BPM,HIGH_BPM))
    return BPM,s.ravel()

def SSR_JOSS(sig,fs,N=None):
    """Removes MA (motion artifacts) from y by means of spectral subtraction
    '''
    Parameters
    sig : Signal object 
        instance of Signal class build with signals from PPG and three axis of an accelerometer
    fs : int
        sampling frequency
    N : int
        resolution :default value (if N is None) is 60*fs to obtain a spectral resolution of 1 BPM
        
    ---------
    Output:
    BPM : numpy array
        axis of BPM frequencies
    SSRsig: numpy array
        cleansed sparsed spectrum (PPG purged of accelerometer artifacts)
    S  : numpy array
        sparse spectra of ppg ,accx,accy,accz
    C  : numpy array
        maxima of sparse spectra of the three accelerometer
    """
    if N is None:
        N = (int)( 60*fs)
    Y = np.zeros((4,len(sig.ppg1)))
    Y[0]=sig.ppg1
    Y[1]=sig.x
    Y[2]=sig.y
    Y[3]=sig.z
   
    #Y must be M x L
    Y = Y.transpose()
    #get the length of Y elements along the second axis
    L = Y.shape[1]
    S = np.zeros((N,L))
    
    for i in range(0,L):
        BPM, SS = SSR(Y[:,i],fs,N)
        #normalize spectra
        #SS = stats.zscore(SS)
        SS = SS / np.max(SS)
        S[:,i] = SS
    aggression = 0.99 # weight for the spectra subtraction
    C = np.zeros(N)
    SSRsig = S[:,0].flatten() # first column is the sparse spectrum of the ppg
    #print("BMP is : ",BPM)
    #looping over all the freq bins:
    for i in range(0,len(BPM)):
        #max of accel signal
        SSacc=np.zeros(3)
        SSacc[0] = S[i,1]
        SSacc[1] = S[i,2]
        SSacc[2] = S[i,3]
       # mag_acc = math.sqrt(SSacc[0]**2+SSacc[1]**2+SSacc[2]**2)
        
       
        C[i] = np.amax(SSacc)
        #C[i] = mag_acc
        #spectral subtraction
        SSRsig[i] = SSRsig[i] - aggression*C[i]
        
       
            
        
    #set values smaller than 1/7-th of the max to 0
    p_max = np.max(SSRsig);
    
    SSRsig[SSRsig<p_max/7] = 0
    return BPM,SSRsig,S,C


def findPksInRange(SSRsig,R):
    """Find peaks of the sparse spectrum of the signal in the range R
    '''
    Parameters:
    ------------
    SSRsig : numpy array
            cleaned sparse spectrum
    R      : numpy array 
            range for searching
    '''
    Output:
    ----------
    pks    : numpy array
                peaks values
    locs   : numpy array
                peak frequencies bins (indices)
    """
    N = len(SSRsig)
    H = np.zeros(N)
    #filter for search range, handles case when search range is negative
    test = np.where(R>0)[0]
    
    #print("test (where): ", test)
    H[R[R>0]]=1 #da guardare
    #Band pass filters spectrum to match search range
    SSRsig = np.multiply(SSRsig,H)
    #return peak values and freq bins
    locs = scipy.signal.find_peaks(SSRsig)[0]
   # print("LOCS IN findPksInRange: ",locs)
    pks  = SSRsig[locs]
    
    return pks,locs

def findClosestPeak(prevLoc,locs):
    """ Finds the closest peak to prevLoc
    '''
    Parameters:
    prevLoc : int 
                index of previous located peak
    locs   : list of peak indices
    
    '''
    Output:
    -------
    curLoc : int
            index of the peak closest to previously located peak
    
    """
    #computes distance to each peak
    #print("prevLoc: ",prevLoc)
    #print("locs found: ",locs)
    dif = np.abs(locs-prevLoc) #DA GUARDARE!!!!!
    #print ("DIF: ",dif)
    index = np.argmin(dif)
    curLoc = locs[index]
    #print("findClosest: ",curLoc)
    return curLoc

def discoverPeak(SSRsig,prevLoc):
    """Searches full spectrum to find a peak
    '''
    Parameters:
    -----------
    SSRsig : numpy array
            sparse spectrum
    prevLoc : int
            index (bin) of previously located HR
    '''
    Output:
    -------
    curLoc : int
            index of discovered peak
    """
    R3 = np.arange(40,241) #full spectrum search
    #R3 = np.arange(prevLoc-50,prevLoc+51)
    _ , locs = findPksInRange(SSRsig,R3)
   # print('LOCS IN DISCOVER: ',locs)
    curLoc = findClosestPeak(prevLoc,locs)
    #print('curLoc in DISCOVER:', curLoc)
    return curLoc

def SPT_JOSS(SSRsig,fs,prevLoc,prevBPM,Trap_count):
    """Searches for the Heart Rate (HR) peak by using previously estimated HR values
    '''
    Parameters:
    ----------
    SSRSig : numpy array
            cleaned spectrum
    fs   : float
            sampling frequency
    prevLoc : int
            index (bin) of previously located HR. initialized as -1
    prevBPM : float
            previous HR in beat per minute. initialized as 0
    '''
    Output
    --------
    curLoc : int
            index (frequency bin) of currently located HR
    curBPM : float
            current HR
    Trap_count: int
            counter that keeps track of the number of times
            no peaks were found, initialized as 0
    """
    
    #print(f"prevLoc is {prevLoc}, prevBPM is {prevBPM}, trap_count is {Trap_count} ")
    
    #parameters of SPT
    delta1 = 10
    delta2 = 20 #frequencies for searching peaks
    
    N = len(SSRsig)
    
    
    #initialization state
    if((prevLoc == -1) and (prevBPM == -1)):
        curLoc = np.argmax(SSRsig)
        curBPM = 60 * curLoc/N*fs
        return curLoc,curBPM,Trap_count
    else:
        #set search range
        #print("R1")
        R1 = np.arange(prevLoc-delta1,prevLoc+delta1+1)
        _, locs = findPksInRange(SSRsig,R1)
       # print(f"R1 prevLoc is {prevLoc}  ")
        #print("R1 LOCS: ",locs)
        numOfPks = len(locs)
        if numOfPks >=1:
            #find closest peak
            curLoc = findClosestPeak(prevLoc,locs)
            curBPM = 60 * curLoc/N*fs
        else: 
            #increase search range
            R2 = np.arange(prevLoc-delta2,prevLoc+delta2+1)
            #print("R2")
            #find peaks in range 2
            pks,locs = findPksInRange(SSRsig,R2)
            #print("R2 locs is ",locs)
            numOfPks = len(locs)
            if numOfPks >=1:
                #find maximum peak
                maxIndex = np.argmax(pks)
                curLoc = locs[maxIndex]
                curBPM = 60 * curLoc/N*fs
            else:
                #choose prev BPM
                curLoc = prevLoc
                curBPM = prevBPM
    
    #validation stage
   # print("VALIDATION")
   # print(f"curLOC is {curLoc} - curBPM is {curBPM}")
    if abs(curBPM-prevBPM)>12:
        Trap_count = Trap_count +1 
        #choose prev BPM
        curLoc = prevLoc
        curBPM = prevBPM
    else:
        if curLoc == prevLoc:
            #print('TRAP')
            Trap_count = Trap_count +1
        else:
            Trap_count = 0
    
    
    
    if(Trap_count > 2):
           # print('DISCOVER')
            #discover peak after three failed peak searchs
            curLoc = discoverPeak(SSRsig,prevLoc)
            curBPM = 60 * curLoc/N*fs
            Trap_count = 0
    
    #print("return curLoc = ",curLoc)
    return curLoc, curBPM, Trap_count



def BandpassFilter(signal, pass_band,fs):
    b, a = sp.signal.butter(2, pass_band, btype='bandpass',fs=fs)
    return sp.signal.filtfilt(b, a, signal)

def RunPulseRateAlgorithm(data_fl, ref_fl):
    # Load data using LoadTroikaDataFile
    ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
    
    # Compute pulse rate estimates and estimation confidence.
    
    #Decimate spectrum by factor 10..with fs = 12.5 Hz (max freq in the spectrum: 375 Hz)
    fsD=12.5 #12.5 Hz as in the Thesis
    ppgD = scipy.signal.decimate(ppg,q=10,zero_phase=True)
    accxD = scipy.signal.decimate(accx,q=10,zero_phase=True)
    accyD = scipy.signal.decimate(accy,q=10,zero_phase=True)
    acczD = scipy.signal.decimate(accz,q=10,zero_phase=True)
    
    #Band pass the signals before using JOSS
    ppgD = BandpassFilter(ppgD,(LOW_BPM_CUT,HIGH_BPM_CUT),fsD)
    accxD = BandpassFilter(accxD,(LOW_BPM_CUT,HIGH_BPM_CUT),fsD)
    accyD = BandpassFilter(accyD,(LOW_BPM_CUT,HIGH_BPM_CUT),fsD)
    acczD = BandpassFilter(acczD,(LOW_BPM_CUT,HIGH_BPM_CUT),fsD)

    window_lengthD =(int)( 8*fsD) #8s window of data
    window_shiftD =(int)( 2*fsD)
    tsD = np.arange(0, len(ppg)*1/fsD, 1/fsD) #time stamp for plotting using the decimated signals
    
    #initialize BPM and location
    prevLoc = -1
    prevBPM = -1
    curBPM = 0
    
    bpm_ref = sp.io.loadmat(ref_fl)['BPM0'] #GROUND TRUTH BMP 
    
    #errors in BPM and confidence
    errors_list = []
    confidence_list = []
    counter=0
    Trap_count = 0 #initialize trap count to 0
    #Let's work in windows
    for i in range(0, len(ppgD)-window_lengthD,window_shiftD):
        #print("PREV BPM IS: ",prevBPM)
        #for every window build the correspondent Signal to be fed into JOSS
        ppg8 = ppgD[i:i+window_lengthD]
        accx8 = accxD[i:i+window_lengthD]
        accy8 = accyD[i:i+window_lengthD]
        accz8 = acczD[i:i+window_lengthD]
        sig8s = Signal(ppg8,accx8,accy8,accz8)
        #print("quality: ",Periodogram(sig8s,60,12.5,50))
        bpm_ref8 = bpm_ref[counter].item()
        BPM,SSRsig,S,C = SSR_JOSS(sig8s,fsD)    
        #print(f"Trap count before calling SPT_JOSS is {Trap_count}")
        curLoc, curBPM, Trap_count = SPT_JOSS(SSRsig,fsD,prevLoc,prevBPM,Trap_count)
        #print(f"Trap count AFTER calling SPT_JOSS is {Trap_count}")
        counter = counter +1
        
        #errors_list.append((bpm_ref8,curBPM))# for debugging
        prevLoc = curLoc
        prevBPM = curBPM
        err=bpm_ref8-curBPM
        
        
        #computing the power spectrum of the signal
        freqs = np.fft.rfftfreq(len(ppg8),1/fsD)
        fft_mag = np.abs(np.fft.rfft(ppg8))
        
        #window for BPM
        window_f = 5 / 60 #from bpm to Hz
        
        hr_freq_window = (freqs>curBPM/60-window_f) & (freqs<curBPM/60+window_f)
        
        sig_power = np.sum(fft_mag[(hr_freq_window)])
        total_power = np.sum(fft_mag)
        #print(f"total power is {total_power}")
        confidence = sig_power/total_power
        
        #print("i: ",str(i),bpm_ref8," - ",curBPM,"error: ",err, "confidence: ",confidence)
        
        errors_list.append(err)
        confidence_list.append(confidence)
        
    
        
    
    # Return per-estimate mean absolute error and confidence as a 2-tuple of numpy arrays.
    errors, confidence =errors_list, confidence_list  # Dummy placeholders. Remove
    return errors, confidence


# In[ ]:




