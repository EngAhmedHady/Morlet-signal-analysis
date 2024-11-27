# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 10:33:32 2024

@author: Ahmed H. Hanfy
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 25})
plt.rcParams["text.usetex"] =  True
plt.rcParams["font.family"] = "Times New Roman"
px = 1/plt.rcParams['figure.dpi']

def plotting(signal: np.ndarray,  ts: np.ndarray,  fs: float,  tw: np.ndarray, 
            cmw: np.ndarray,  cmwFFT: np.ndarray, ssFFT: np.ndarray, 
            nconv: int,  conv: np.ndarray, conv_recons: np.ndarray, 
            units: str = 'mm') -> None:
    """
    Visualize the wavelet generation, convolution process, and filtered signal analysis.

    Parameters:
        - **signal (np.ndarray)**: Input signal in the time domain.
        - **ts (np.ndarray)**: Time vector corresponding to the input signal.
        - **fs (float)**: Sampling frequency of the signal.
        - **tw (np.ndarray)**: Time vector for the wavelet, centered around zero.
        - **cmw (np.ndarray)**: Complex Morlet wavelet.
        - **cmwFFT (np.ndarray)**: Fourier transform of the wavelet.
        - **ssFFT (np.ndarray)**: Fourier transform of the input signal.
        - **nconv (int)**: Convolution length (signal + wavelet - 1).
        - **conv (np.ndarray)**: Convolution result in the frequency domain.
        - **conv_recons (np.ndarray)**: Filtered signal in the time domain.
        - **units (str, optional)**: Units for labeling the plots (default: 'mm').

    Visualizations:
        1. **Original Signal**: Plots the input signal in the time domain.
        2. **Wavelet Visualization**:
            - Time-domain representation of the complex Morlet wavelet.
            - 3D visualization showing real and imaginary components of the wavelet.
        3. **Frequency-Domain Analysis**:
            - Fourier transform of the input signal.
            - Fourier transform of the wavelet.
            - Fourier transform of the convolution result.
        4. **Complex Representation of Filtered Signal**:
            - 3D visualization of the filtered signal in the complex domain.
        5. **Filtered Signal Analysis**:
            - Comparison of the original and filtered signal in the time domain.
            - Power and phase plots of the filtered signal.
        

    Key Equations:
        - **Wavelet Function**:
          \[ \psi(t) = e^{2 \pi i f_w t} e^{-\frac{t^2}{2 \sigma^2}} \]
        - **Frequency-Domain Convolution**:
          \[ \text{conv}(f) = \mathcal{F}(\text{signal}) \cdot \mathcal{F}(\psi) \]
        - **Reconstructed Signal**:
          \[ \text{signal}_{\text{filtered}} = \mathcal{F}^{-1}(\text{conv}(f)) \]

    Example:
        >>> plotting(signal, ts, fs, tw, cmw, cmwFFT, ssFFT, nconv, conv, conv_recons)
    """
    # Frequancy domain = original frquancy / 2,
    # Note: the signal length is total number of convolution length,
    #       it will be cutted it later
    # Frequency domain range (up to Nyquist frequency)
    f_line = np.linspace(0, int(fs/2), np.floor(nconv/2).astype(int) + 1)
    
    # original signal plot
    fig,ax = plt.subplots(figsize=(20, 7))
    ax.plot(ts, signal)
    ax.set_title('Original Signal')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel(f'Amplitude ({units})')
    ax.set_xlim([0,0.5]) # limited to half second only
    
    # complex representation of the wavelet
    ax = plt.figure(figsize=(15, 15)).add_subplot(projection='3d')
    ax.plot(tw,np.real(cmw), -1, '--k', alpha=0.2)
    ax.plot(tw, np.ones(len(tw)), np.imag(cmw), '--k', alpha=0.2)
    ax.plot(-np.ones(len(tw)), np.real(cmw), np.imag(cmw), '--k', alpha=0.2)
    ax.plot(tw,np.real(cmw), np.imag(cmw), lw=3)
    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('Time (sec)', labelpad=35)
    ax.set_ylabel('Real amplitude', labelpad=35)
    ax.set_zlabel('Imaginary amplitude', labelpad=40)
    ax.set_box_aspect(None, zoom=0.85)
    ax.set_zticks(np.arange(-1,1.1,0.25),minor=False)
    ax.set_zticklabels(ax.get_zticks(), fontdict={'ha':'left'}, fontsize=25)
    ax.set_xticklabels(ax.get_xticks(),fontsize=25)
    ax.set_yticklabels(ax.get_yticks(), fontsize=25)
    # ax.set_title('Wavelet')
    # for rotating the view
    # ax.view_init(elev=90, azim=0, roll=0)
    
    # Frequency-domain convolution plots
    # Convelution process (Original signal FFT, morlet FFT, the convoluted signal)
    fig,ax = plt.subplots(3, 1, figsize=(16, 22))
    fig.tight_layout(h_pad=3)
    ax[0].plot(f_line, 2*np.abs(ssFFT)[:len(f_line)])
    ax[0].set_title('Original Signal FFT')
    ax[0].set_ylabel(units)
    ax[1].plot(f_line, np.abs(cmwFFT)[:len(f_line)])
    ax[1].set_title('Wavelet FFT')
    ax[1].set_ylabel('Amplitude')
    ax[2].plot(f_line, 2*np.abs(conv)[:len(f_line)])
    ax[2].set_title('Convolution FFT')
    ax[2].set_ylabel(units)
    for i in ax: 
        i.set_xlim([0,750])
        i.set_xlabel('Frequancy (Hz)')
        i.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw=1.5)
        i.minorticks_on()
        i.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)
        
    # Original and filtered signals    
    fig,ax = plt.subplots(figsize=(20, 7))
    ax.plot(ts,signal, label='Original Signal')
    ax.plot(ts,np.real(conv_recons), label='Filtered Signal')
    ax.set_xlim([0,0.5])
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel(f'Amplitude ({units})')
    ax.legend()
    
    # complex representation of the filtered signal
    Plot_recon = conv_recons / np.amax(abs(conv_recons))
    ax = plt.figure(figsize=(15,15)).add_subplot(projection='3d')
    ax.plot(ts[0:750], np.real(Plot_recon[0:750]),-1,'--k', alpha=0.2)
    ax.plot(ts[0:750], np.ones(750), np.imag(Plot_recon)[0:750], '--k', alpha=0.2)
    ax.plot(np.zeros(750), np.real(Plot_recon)[0:750], np.imag(Plot_recon)[0:750], '--k', alpha=0.2)
    ax.plot(ts[0:750], np.real(Plot_recon)[0:750], np.imag(Plot_recon)[0:750])
    ax.set_xlim([0,0.5])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_xlabel('Time (sec)', labelpad=20)
    ax.set_ylabel(f'Amplitude ({units})', labelpad=20)
    ax.set_zlabel('imaginary amplitude', labelpad=35)
    ax.set_box_aspect(None, zoom=0.9)
    ax.set_zticks(np.arange(-1,1.1,0.25), minor=False)
    ax.set_zticklabels(ax.get_zticks(), fontdict={'ha':'left'})
    
    # Filtered signal analysis
    fig,ax = plt.subplots(3, 1, figsize=(16, 22))
    fig.tight_layout(h_pad=3)
    ax[0].plot(ts, np.real(conv_recons))
    ax[0].plot(ts, signal)
    ax[0].set_title('Filtered signal')
    ax[0].set_ylabel(f'Amplitude ({units})')
    ax[1].plot(ts,np.real(conv_recons))
    ax[1].plot(ts,np.abs(conv_recons))
    ax[1].set_title('Magnitude signal (power)')
    ax[1].set_ylabel(f'Power amplitude ({units}$^2$)')
    ax[2].plot(ts,np.angle(conv_recons))
    ax[2].set_title('Signal phase')
    ax[2].set_ylabel('Phase (rad)')
    for i in ax: 
        i.set_xlim([0, 0.5])
        i.set_xlabel('Time (sec)')
        i.grid(True, which='major', color='#D8D8D8', linestyle='-', alpha=0.3, lw = 1.5)
        i.minorticks_on()
        i.grid(True, which='minor', color='#D8D8D8', linestyle='-', alpha=0.2)

def WaveletGeneration(signal: np.ndarray, nconv: int, tw: np.ndarray, ts: np.ndarray,
                      fw: float, sigma: float,  fs: float, 
                      ploting: bool = False, units: str = 'mm') -> np.ndarray:
    """
   Generate a wavelet filter and apply it to the input signal using frequency-domain convolution.

   Parameters:
       - **signal (np.ndarray)**: Input signal array, shape (n_samples,).
       - **nconv (int)**: Length of the convolution (original signal length + wavelet length - 1).
       - **tw (np.ndarray)**: Time vector for the wavelet function.
       - **ts (np.ndarray)**: Time vector for the original signal.
       - **fw (float)**: Frequency of the wavelet.
       - **sigma (float)**: Standard deviation of the Gaussian envelope in the wavelet.
       - **fs (float)**: Sampling frequency of the signal in Hz.
       - **ploting (bool, optional)**: Whether to generate diagnostic plots during processing (default: False).
       - **units (str, optional)**: Units for labeling plots (default: 'mm').

   Returns:
       - **np.ndarray**: Filtered signal (same length as the input signal).
       
   Equations:
        - **Wavelet Function (Complex Morlet Wavelet)**:
          \[ \psi(t) = e^{2 \pi i f_w t} e^{-\frac{t^2}{2 \sigma^2}} \]
          where:
          - \(f_w\) is the center frequency of the wavelet.
          - \(\sigma\) is the standard deviation of the Gaussian envelope.
        
        - **Frequency-Domain Convolution**:
          \[ \text{conv}(f) = \mathcal{F}(\text{signal}) \cdot \mathcal{F}(\psi) \]
          where:
          - \(\mathcal{F}\) is the Fourier transform.
          - \(\psi\) is the wavelet function.
        
        - **Reconstructed Signal**:
          \[ \text{signal}_{\text{filtered}} = \mathcal{F}^{-1}(\text{conv}(f)) \]
          where \(\mathcal{F}^{-1}\) is the inverse Fourier transform.
        
        - **Normalization**:
          The wavelet's FFT is normalized to ensure the convolution does not scale the signal:
          \[ \mathcal{F}(\psi) \leftarrow \frac{\mathcal{F}(\psi)}{\max(\mathcal{F}(\psi))} \]

   Example:
       >>> import numpy as np
       >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))
       >>> fs = 500
       >>> nconv = len(signal) + 100 - 1
       >>> tw = np.arange(-1, 1, 1 / fs)
       >>> ts = np.linspace(0, 1, len(signal))
       >>> fw = 5
       >>> sigma = 7 / (2 * np.pi * fw)
       >>> filtered_signal = WaveletGeneration(signal, nconv, tw, ts, fw, sigma, fs)
       >>> print(filtered_signal)

   .. note ::
       - The function uses frequency-domain convolution for computational efficiency.
       - Convolution is performed in the Fourier domain and then transformed back using the inverse Fourier transform.
       - Zero-padding is applied to maintain the original signal's dimensions.

   """
    # Generate the wavelet function (complex Morlet wavelet)
    cmw = np.exp(np.pi*fw*tw*2j)*np.exp((-tw**2)/(2*sigma**2))
    # Apply convolution in frequancy domain (for optimal operation)
    # Note: "nconv" used to make the output signal with additional zero borders 
    #       for convelution
    cmwFFT = np.fft.fft(cmw,nconv)  # ... FFT of the wavelet
    cmwFFT /= max(cmwFFT)  # ............ Normalize to maintain original signal units
    ssFFT = np.fft.fft(signal,nconv)  # . FFT of the signal
    conv =  ssFFT * cmwFFT  # ........... Perform convolution in the frequency domain
    
    # Reconstruct the filtered signal using inverse FFT
    hw = np.floor(len(cmw)/2).astype(int)+1  # Half-length of the wavelet
    conv_recons = np.fft.ifft(conv)  # ....... Inverse FFT to reconstruct signal       
    # Extract the signal segment matching the original
    conv_recons = conv_recons[hw-1:-(hw-2)]
    
    # Optional plotting for diagnostic purposes
    if ploting:
        plotting(signal,ts,fs,
                 tw,cmw,cmwFFT,ssFFT,
                 nconv, conv,conv_recons,
                 units)
    
    return conv_recons

# def morlet(signal, fs, fw = 50, n_cycles = 7, review = False, units = 'mm'):
def morlet(signal: np.ndarray, 
           fs: float, fw: float | list[float] = 50, n_cycles: int | list[int] = 7, 
           review: bool | list[int] = False, units: str = 'mm') -> np.ndarray:
    """
    Perform a Morlet wavelet transform on the given signal for specified frequencies and parameters.

    Parameters:
        - **signal (np.ndarray)**: Input signal array, shape (n_samples,).
        - **fs (float)**: Sampling frequency in Hz.
        - **fw (float | list[float], optional)**: Frequency or list of frequencies for wavelet transform (default: 50).
        - **n_cycles (int | list[int], optional)**: Number of cycles in the wavelet. Can be an integer or a list of values (default: 7).
        - **review (bool | list[int], optional)**: Whether to preview specific frequency intervals.
          If `True`, plots the entire frequency range; if a list, specify start and end indices for plotting.
        - **units (str, optional)**: Units for plotting (default: 'mm').

    Returns:
        - **np.ndarray**: Time-frequency representation of the signal with shape (len(fw), n_samples) or (n_samples,).

    Example:
        >>> import numpy as np
        >>> signal = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 500))
        >>> fs = 500
        >>> fw = [5, 10, 20]
        >>> tf = morlet(signal, fs, fw, n_cycles=7, review=False)
        >>> print(tf.shape)

    .. note ::
        - For single frequency (`fw` is float), the output is the reconstructed signal.
        - For multiple frequencies, the output is a 2D array where each row corresponds to one frequency.
        - If `review` is a list, the specified range of frequencies is plotted during the transform.

    """
    # n_cycles is the Number of cycles, Ex: n_cycles = 3  => --v/^\v--
    
    # original signal parameters
    nShoots = signal.shape[0]  # .............. Number of samples in the signal
    T = nShoots/fs  # ......................... Total duration of the signal in seconds
    ts = np.linspace(0, T, nShoots)  # ........ Time vector
    
    # wavelet parameters
    tw = np.arange(-1,1,1/fs)  # .............. Time vector for wavelet generation
    
    # Final convolution length (signal length + wavelet length - 1)
    nconv = nShoots+len(tw)-1
    
    
    if hasattr(fw, "__len__"):  # Multiple frequencies
        n_fw = len(fw)
        # Handle review parameters
        if hasattr(review, "__len__"): # If review is a list, define plotting range
            review.sort()
            start, end = review
            plotingInterval = abs(end-start)
            ploting = plotingInterval > 0
        else: 
            start, end = 0, 0
            ploting = False
            
        # Handle number of cycles for each frequency   
        if not hasattr(n_cycles, "__len__"): # If single value, broadcast to match fw
            n_cycles = n_cycles * np.ones(n_fw)

        # Standard deviation of the Gaussian kernel
        sigma = n_cycles / (2*np.pi*fw)
        tf = np.zeros([n_fw, nShoots])  # Initialize time-frequency representation 
        for i in range(n_fw):  # Process each frequency
            Plot = ploting and start <= i < end  # Enable plotting for the specified range
            conv_recons = WaveletGeneration(signal, nconv, tw, ts,
                                            fw[i], sigma[i], fs,
                                            ploting = Plot, units = 'mm')
            
            tf[i,:] = np.abs(conv_recons)**2  # Power of the wavelet transform
    else: # Single frequency
        sigma = n_cycles / (2*np.pi*fw)
        tf = WaveletGeneration(signal, nconv, tw, ts, fw, sigma, fs, 
                               ploting = review, units = 'mm')
    return tf