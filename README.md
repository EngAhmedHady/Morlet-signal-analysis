# Morlet Continuous Wavelet Transform (CWT)

## Introduction
To analyze the temporal evolution of spectral signatures in fluctuating flow fields, 
the **Continuous Wavelet Transform (CWT)** is utilized. 
This transformative approach decomposes a signal into highly localized wavelets, 
enabling detailed inspection in both time and frequency domains 
[[1](#1), [2](#2)]. By employing **complex Morlet wavelets**, 
the method provides a powerful representation of time-frequency dynamics. The **Morlet wavelet**, 
characterized by the modulation of a Gaussian envelope with a complex sinusoidal function,
 is particularly well-suited for analyzing non-stationary signals with varying spectral content.

### Key Features of CWT:
1. **Time-Frequency Localization**: Wavelets allow for the simultaneous analysis of time and frequency components, capturing transient features in the data.
2. **Scalability**: The method accommodates both single-frequency and multi-frequency analyses.
3. **Robust Convolution Framework**: Using frequency-domain convolution, the method ensures computational efficiency while preserving signal fidelity.

### Morlet Wavelet Definition
The Morlet wavelet function $\psi(t)$ is defined as:

$$\psi(t) = e^{2 \pi i f_w t} e^{-\frac{t^2}{2 \sigma^2}}$$

where:
- $f_w$: Center frequency of the wavelet.
- $\sigma$: Standard deviation of the Gaussian envelope, determined by the number of cycles ($n_{\text{cycles}}$) as:

$$\sigma = \frac{n_{\text{cycles}}}{2 \pi f_w}$$

### Key Equations
1. **Frequency-Domain Convolution**:

$$\text{conv}(f) = \mathcal{F}(\text{signal}) \cdot \mathcal{F}(\psi)$$
   
   where $\mathcal{F}$ denotes the Fourier transform.

2. **Reconstructed Signal**:

$$\text{signal}_{\text{filtered}} = \mathcal{F}^{-1}(\text{conv}(f))$$
   
   where $\mathcal{F}^{-1}$ is the inverse Fourier transform.

3. **Normalization**:
   To preserve the amplitude integrity, the wavelet's Fourier transform is normalized as:
   
$$\mathcal{F}(\psi) \leftarrow \frac{\mathcal{F}(\psi)}{\max(\mathcal{F}(\psi))}$$

### Example Usage
The library includes implementations for:
- **Single-Frequency Analysis**: Focused wavelet transform at a specific frequency.
- **Multi-Frequency Analysis**: Computation of the time-frequency representation for a range of frequencies.
- **Diagnostic Plots**: Visualization of original, filtered, and reconstructed signals, along with frequency-domain representations.

## Installation
Simply include the `morlet.py` file in your project and import the functions as needed:
```python
import morlet as wl
```

## Functions Overview
### ``morlet``
Performs a Morlet wavelet transform on the input signal. Supports both single-frequency and multi-frequency analyses.

#### Parameters:
   - **``signal``** (np.ndarray): Input signal array.
   - **``fs``** (float): Sampling frequency in Hz.
   - **``fw``** (float | list[float], optional): Target frequency (or list of frequencies) for analysis.
   - **``n_cycles``** (int | list[int], optional): Number of cycles for each wavelet. Defaults to 7.
   - **``review``** (bool | list[int], optional): Whether to preview results. Can specify a frequency range for plotting.
   - **``units``** (str, optional): Units for the signal (default: 'mm').
#### Returns:
   - np.ndarray: The transformed signal(s) in time-frequency space.

### ``WaveletGeneration``
Generates a complex Morlet wavelet and applies it to the input signal using frequency-domain convolution.

#### Parameters:
   - **``signal``**, **``nconv``**, **``tw``**, **``ts``**, **``fw``**, **``sigma``**,
    **``fs``**, **``ploting``**, **``units``**...
#### Returns:
   - np.ndarray: Filtered signal reconstructed via inverse FFT.

#### Note:
    This function is internally called by morlet.

### Example
Import the necessary libraries and adjust plotting parameters
```python
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import morlet as wl
>>> from matplotlib.ticker import MaxNLocator
>>> from matplotlib.colors import BoundaryNorm
>>> plt.rcParams.update({'font.size': 30})
>>> plt.rcParams["text.usetex"] =  False
>>> plt.rcParams["font.family"] = "Times New Roman"
```
Import signal file using pandas dataframe for easy data manipulation, note the data sampled in 1500Hz with a 60Hz imposed signal. 
```python
>>> file_path = 'signal.txt'
>>> signal = pd.read_csv(fr'{file_path}', header=None)
>>> n_samples = len(signal)
```

The input parameters are: 
```python
# Signal info.
>>> fs = 1500
>>> T = n_samples/fs
>>> ts = np.linspace(0, T, n_samples)

# Wavelet prams.
>>> transition_steps = 100
>>> n_cycles = 64
>>> fw = np.linspace(1,int(fs/2),transition_steps)
```
Then Compute and plot the CWT.
```python
>>> cwtm = wl.morlet(signal[1], fs, n_cycles=n_cycles, fw=fw, units='mm')

# Visualise morlet CWT
>>> fig, ax = plt.subplots(figsize=(20, 10))
>>> levels = MaxNLocator(nbins=n_color_levels).tick_values(0, maxvalue)
>>> cmap = plt.colormaps['jet']
>>> norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
>>> im = ax.contourf(ts, fw, cwtm, levels=levels, cmap=cmap, norm=norm)
>>> levels = MaxNLocator(nbins=n_levels).tick_values(0, maxvalue)
>>> imk = ax.contour(im, colors='k', levels=levels, alpha=np.linspace(1, 0.2,n_levels+1))
>>> cbar = fig.colorbar(im, ax=ax, ticks=np.around(np.linspace(0, maxvalue, 5), 3))
>>> ax.set_xlabel("Time (sec)")
>>> ax.set_ylabel("Frequency (Hz)")
>>> ax.set_ylim([1, 750])
# Location of imposed signal
>>> ax.hlines(60, 0, T, 'w', ls='--', alpha=0.5)
>>> ax.text(T-0.45, 60+5, '60Hz', fontsize=22, alpha=0.5, color='w')
```


## References
<a id="1"></a>1. Basley, J., Perret, L., & Mathis, R. (2018). Spatial modulations of kinetic energy in the roughness sublayer. Journal of Fluid Mechanics, 850, 584–610. DOI: [10.1017/jfm.2018.458](http://dx.doi.org/10.1017/jfm.2018.458)
   
<a id="2"></a>2. Russell, B., & Han, J. (2016). Jean Morlet and the continuous wavelet transform. CREWES Research Report, 28, 115.

