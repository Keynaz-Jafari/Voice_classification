import sounddevice as sd
from scipy.io.wavfile import write


def record_audio(sample_rate=44100):
    output_file = input("Enter the file name (e.g., 'output.wav'): ")
    duration = int(input("Enter the duration of the recording (in seconds): "))


    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")

    # Save the audio data to a .wav file
    write(output_file, sample_rate, audio_data)
    print(f"Audio saved to {output_file}")



if __name__ == "__main__":
    record_audio()
