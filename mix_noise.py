import argparse
import numpy as np
import soundfile as sf

def mono(audio):
    if audio.ndim == 2:
        return audio.mean(axis=1) #if audio has two dimensions
    else:
        return audio

def fit_length(audio, length):
    if len(audio) == 0:
        raise ValueError("Construction noise file is empty")
    repeats = int(np.ceil(length/len(audio))) #how many repetitions are needed to reach the requested length
    return np.tile(audio, repeats)[:length]

def rms(audio): #average energy level of an audio signal
    return np.sqrt(np.mean(audio**2) + 1e-12)

def main():
    parser = argparse.ArgumentParser()
    #add args to be parsed below
    parser.add_argument("--clean", default="clean.wav", help="clean speech WAV file")
    parser.add_argument("--construction", default="construction.wav", help="construction noise WAV file")
    parser.add_argument("--output", default="dirty_construction_0dB.wav", help="Output noisy WAV file")
    parser.add_argument("--snr", type=float, default=0.0, help="target SNR in decibels")
    #parser.add_argument("__type", type=dtype, default)
    #read all arguments provided above
    args = parser.parse_args()
    clean, clean_sr = sf.read(args.clean, dtype="float32") #clean_sr maybe 16000 samples per second
    construction, noise_sr = sf.read(args.construction, dtype="float32")
    if clean_sr != noise_sr:
        raise ValueError(f"Sample-rate mismatch: clean={clean_sr}, noise={noise_sr}")

    clean=mono(clean).astype("float32")

    construction = mono(construction).astype("float32")
    construction = fit_length(construction, len(clean))

    clean_rms = rms(clean)
    #if snr = 10db, then speech is 10db stronger than the noise

    target_noise_rms = clean_rms / (10**(args.snr/20))
    current_noise_rms = rms(construction)

    construction*= target_noise_rms / current_noise_rms

    noisy = clean+construction

    peak = np.max(np.abs(noisy))

    if peak>0.99:
        noisy*=0.99/peak

    sf.write(args.output,noisy,clean_sr)


    measured_snr = 20 * np.log10(rms(clean) / rms(construction)) #calculate actual snr achieved after mixing

    #print the summary of generated test file
    print(f"Wrote: {args.output}")
    print(f"Sample rate: {clean_sr} Hz")
    print(f"Duration: {len(clean) / clean_sr:.2f} seconds")
    print(f"Measured SNR: {measured_snr:.2f} dB")


#__main__ entry point
if __name__ == "__main__":
    main()
