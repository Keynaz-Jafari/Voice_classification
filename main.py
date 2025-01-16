import numpy as np
import wave
import matplotlib.pyplot as plt

###############################################################################

#function haye khodam

def dft(signal): #transforms signals to frequency domain

    N = len(signal)
    dft_result = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            dft_result[k] += signal[n] * np.exp(-2j * np.pi * k * n / N)
    return dft_result


def idft(frequency_signal): #transforms signal back to time domain
    N = len(frequency_signal)
    idft_result = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            idft_result[n] += frequency_signal[k] * np.exp(2j * np.pi * k * n / N)
    return idft_result / N


def compute_energy(signal):

    return np.sum(np.abs(signal) ** 2)


###############################################################################


# 1) Load Audio Files and Extract Discrete Signals

def get_discrete_signal(file_path, resample_rate=None):

   # Loads a .wav file and returns a discrete signal, its time values, and the original sampling rate.

    with wave.open(file_path, 'r') as wav_file:
        original_sample_rate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        duration = n_frames / original_sample_rate
        n_channels = wav_file.getnchannels()

        raw_data = wav_file.readframes(n_frames)
        # Assuming 16-bit PCM
        signal = np.frombuffer(raw_data, dtype=np.int16)

        # If stereo, take just one channel
        if n_channels > 1:
            signal = signal[::n_channels]

        # Create time array
        time_values = np.linspace(0, duration, len(signal), endpoint=False)

        # Resample if needed
        if resample_rate and resample_rate != original_sample_rate:
            from scipy.signal import resample
            num_samples = int(duration * resample_rate)
            signal = resample(signal, num_samples)
            time_values = np.linspace(0, duration, len(signal), endpoint=False)

    return signal, time_values, original_sample_rate

###############################################################################
# 2) Filter Signals in the Frequency Domain (Using NumPy FFT)

def filter_signal_frequency_domain(signal, sample_rate, low_freq=50, high_freq=5000): # remove noise

    N = len(signal)

    # 1. Compute DFT using numpy
    X = np.fft.fft(signal)

    #dft without library
    #X = dft(signal)

    # Frequency resolution
    freq_resolution = sample_rate / N

    # 2) Filter in the frequency domain
    for k in range(N):
        freq_k = k * freq_resolution
        if freq_k < low_freq or freq_k > high_freq:
            X[k] = 0

    # 3) Compute inverse FFT
    filtered_signal_complex = np.fft.ifft(X)

    #idft bedune library
    #filtered_signal_complex = idft(X)


    # Return real part (assuming original signal is real)
    filtered_signal = np.real(filtered_signal_complex)

    return filtered_signal

###############################################################################
# 3) Compute Signal Energy

def compute_energy(signal):

    return np.sum(np.abs(signal) ** 2)
    # energy bedune library
    #return compute_energy(signal)


###############################################################################
# 4) Classify Voice Signals
def classify_voice(energy_value, avg_energy_male, avg_energy_female):
    """
    Classify a voice signal based on comparing its energy to the
    average energies of male and female signals.

    Args:
        energy_value (float): Energy of the new signal.
        avg_energy_male (float): Average energy of male reference signals.
        avg_energy_female (float): Average energy of female reference signals.

    Returns:
        label (str): "Male" or "Female"
    """
    # Simple approach: whichever it is closest to
    diff_male = abs(energy_value - avg_energy_male)
    diff_female = abs(energy_value - avg_energy_female)

    if diff_male < diff_female:
        return "Male"
    else:
        return "Female"

###############################################################################

if __name__ == "__main__":

    male_file_path = "/Users/keinaz/PycharmProjects/signalProject/baba"
    female_file_path = "/Users/keinaz/PycharmProjects/signalProject/keinaz"

    male_signal, male_time, male_sr = get_discrete_signal(male_file_path)

    female_signal, female_time, female_sr = get_discrete_signal(female_file_path)

    # 2) Filter Signals (Remove Noise outside 50-5000 Hz)

    male_filtered = filter_signal_frequency_domain(male_signal, male_sr, low_freq=50, high_freq=5000)
    female_filtered = filter_signal_frequency_domain(female_signal, female_sr, low_freq=50, high_freq=5000)

    # 3) Compute Energies

    male_energy = compute_energy(male_filtered)
    female_energy = compute_energy(female_filtered)

    # For a more robust scenario with multiple samples:
    #   avg_energy_male = average(energies_of_multiple_male_files)
    #   avg_energy_female = average(energies_of_multiple_female_files)
    # Here, we just use single-file energies for demonstration.
    avg_energy_male = male_energy
    avg_energy_female = female_energy

    print(f"Reference Male Energy: {avg_energy_male}")
    print(f"Reference Female Energy: {avg_energy_female}")

    # 4) Classify a New Voice Recording

    new_file_path = "/Users/keinaz/PycharmProjects/signalProject/nameeeee"
    new_signal, new_time, new_sr = get_discrete_signal(new_file_path)

    new_filtered = filter_signal_frequency_domain(new_signal, new_sr, low_freq=50, high_freq=5000)

    new_energy = compute_energy(new_filtered)
    print(f"New Signal Energy: {new_energy}")

    # Classify based on which reference energy is closer
    result = classify_voice(new_energy, avg_energy_male, avg_energy_female)

    print(f"Classification Result: {result}")


    # Visualization for the input signal to classify

    # Compute FFT for the input signal and its filtered version
    new_fft = np.fft.fft(new_signal)
    new_filtered_fft = np.fft.fft(new_filtered)
    freqs_new = np.fft.fftfreq(len(new_signal), 1 / new_sr)

    # Create subplots
    fig, axs = plt.subplots(3, 1, figsize=(7, 10))

    # 1. Plot the input signal in the time domain
    axs[0].plot(new_time, new_signal, label="Original Signal")
    axs[0].plot(new_time, new_filtered, label="Filtered Signal", color='r', alpha=0.7)
    axs[0].set_title("Input Signal Before and After Filtering (Time Domain)")
    axs[0].set_xlabel("Time [s]")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()

    # 2. Plot the input signal in the frequency domain (FFT before filtering)
    axs[1].plot(freqs_new[:len(freqs_new) // 2], np.abs(new_fft[:len(freqs_new) // 2]), label="Original FFT")
    axs[1].set_title("Input Signal in Frequency Domain (Before Filtering)")
    axs[1].set_xlabel("Frequency [Hz]")
    axs[1].set_ylabel("Magnitude")
    axs[1].legend()

    # 3. Plot the filtered signal in the frequency domain (FFT after filtering)
    axs[2].plot(freqs_new[:len(freqs_new) // 2], np.abs(new_filtered_fft[:len(freqs_new) // 2]), label="Filtered FFT", color='r')
    axs[2].set_title("Filtered Signal in Frequency Domain (After Filtering)")
    axs[2].set_xlabel("Frequency [Hz]")
    axs[2].set_ylabel("Magnitude")
    axs[2].legend()

    # Adjust layout and show plots
    plt.tight_layout()
    plt.show()
