
## *Voice Classification System*

### **Overview**

This system is designed to classify voice signals as either male or female using their energy and frequency characteristics. The main workflow involves:

- Loading audio signals and converting them into discrete data.
- Using mathematical transformations (DFT and IDFT) for frequency domain analysis.
- Filtering out noise to retain essential signal components.
- Computing the energy of the signals for comparison and classification.

---

### Key Functions

1. **`discrete fourier transform (dft)`** 
    - Converts a time-domain signal into the frequency domain.
    - Analyzes the signal's frequency components by summing contributions of all time samples with a mathematical formula. Useful for identifying the dominant frequencies in the signal.
    
    ```python
    def dft(signal): #transforms signals to frequency domain
    
        N = len(signal)
        dft_result = np.zeros(N, dtype=complex)
        for k in range(N):
            for n in range(N):
                dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
        return dft_result
    ```
    
2. **`inverse discrete fourier transform (idft)`** 
- Converts a frequency-domain signal back into the time domain.
- Reconstructs the original signal after processing in the frequency domain. This allows for filtered signals to be converted back for further time-domain analysis.

```python
def idft(frequency_signal): #transforms signal back to time domain
    N = len(frequency_signal)
    idft_result = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            idft_result[n] += frequency_signal[k] * np.exp(2j * np.pi * k * n / N)
    return idft_result / N
```

1. **`signal energy computation`**
    - Calculates the total energy by summing the squares of the signal's amplitude values. Energy is critical for comparing signals and classifying them based on predefined energy averages for male and female voices.
    
    ```python
    def compute_energy(signal):
    
        return np.sum(np.abs(signal) ** 2)
    ```
    

---

## **Implementation Steps**

### **1. Load Audio Signals**

- Save voice recordings as `.wav` files.
- Extract discrete signals using provided Python function.

### **2. Implement DFT and IDFT**

- Custom functions for DFT and IDFT. (or use numpy library fft and ifft)

### **3. Filter Signals in the Frequency Domain**

- Transform signals into the frequency domain using DFT.
- Remove noise by zeroing out frequencies outside 50–5000 Hz.
- Transform back to the time domain using IDFT.

### **4. Compute Signal Energy**

- compute signal energy.
- Compare filtered signal energy with reference values. ( **reference values are the average energy of 5 male and 5 female inputs)**

### **5. Classify Voice Signals**

- Process new voice signals.
- Classify as "Male" or "Female" by comparing their energy to reference averages.

### **5. Plot the output**

- Plot the input signal before and after applying a filter.
- show the results of the Discrete Fourier Transform (DFT) when the signal is in the frequency domain.

---

**You can create a .wav file using wav.py code.**

For running the program, you need to install these python libraries: 
- numpy
- scipy
- matplotlib
- ffmpeg
- sounddevice
