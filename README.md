# Wavelet Signal Analysis and Continuous Wavelet Transform (CWT) Library

## Introduction
To analyze the temporal evolution of spectral signatures in fluctuating flow fields, the **Continuous Wavelet Transform (CWT)** is utilized. This transformative approach decomposes a signal into highly localized wavelets, enabling detailed inspection in both time and frequency domains [1][1], [2][2]. By employing **complex Morlet wavelets**, the method provides a powerful representation of time-frequency dynamics. The **Morlet wavelet**, characterized by the modulation of a Gaussian envelope with a complex sinusoidal function, is particularly well-suited for analyzing non-stationary signals with varying spectral content.

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
Simply include the `wavelet_analysis.py` file in your project and import the functions as needed:
```python
from wavelet_analysis import morlet
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

## References
   [1]: Basley, J., Perret, L., & Mathis, R. (2018). Spatial modulations of kinetic energy in the roughness sublayer. Journal of Fluid Mechanics, 850, 584–610. DOI: [10.1017/jfm.2018.458](http://dx.doi.org/10.1017/jfm.2018.458)
   [2]: ussell, B., & Han, J. (2016). Jean Morlet and the continuous wavelet transform. CREWES Research Report, 28, 115.

