import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scipy.signal import periodogram

# Load FITS data
def load_fits(file_path):
    with fits.open(file_path) as hdul:
        hdul.info()  # Display structure of the FITS file
        print("Columns in the first data extension:")
        print(hdul[1].columns)  # Inspect columns in the first data extension
        data = hdul[1].data  # Adjust HDU index if needed
    return data

# Inspect DataArray contents
def inspect_data_array(data, num_rows=5):
    data_array = data['DataArray']
    for i in range(min(num_rows, len(data_array))):
        print(f"Row {i} DataArray:", data_array[i])

# Generate Light Curve by summing counts across all energy channels
def generate_light_curve(data):
    try:
        time = data['TIME']  # Time column exists
        data_array = data['DataArray']  # DataArray column
        # 'DataArray' is of type '2048B' => 2048 unsigned bytes
        # Convert 'DataArray' to numpy array of shape (n_time, 2048)
        counts_array = np.array([list(row) for row in data_array])  # Shape (n_time, 2048)
        # Sum counts across all channels for each time bin
        counts = counts_array.sum(axis=1)
        plt.figure(figsize=(10,6))
        plt.plot(time, counts, color='blue')
        plt.title('Light Curve')
        plt.xlabel('Time (s)')
        plt.ylabel('Counts')
        plt.grid(True)
        plt.show()
        return time, counts
    except KeyError as e:
        print(f"KeyError: {e}. Available columns: {data.columns.names}")
        raise

# Generate Power Density Spectrum
def generate_pds(counts, time_step):
    freqs, power = periodogram(counts, fs=1 / time_step)
    plt.figure(figsize=(10,6))
    plt.loglog(freqs, power, color='red')
    plt.title('Power Density Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.grid(True)
    plt.show()
    return freqs, power

# Generate Hardness-Intensity Diagram (HID)
def generate_hid(data, energy_bands):
    try:
        data_array = data['DataArray']  # DataArray column
        # Convert 'DataArray' to numpy array of shape (n_time, 2048)
        counts_array = np.array([list(row) for row in data_array])  # Shape (n_time, 2048)
       
        # Define energy bands (channel indices)
        hard_low, hard_high, soft_low, soft_high = energy_bands

        # Check if the energy band indices are within the range
        if hard_high > counts_array.shape[1] or soft_high > counts_array.shape[1]:
            raise ValueError("Energy band indices exceed number of channels.")

        # Sum counts in hard and soft bands
        hard_band = counts_array[:, hard_low:hard_high].sum(axis=1)
        soft_band = counts_array[:, soft_low:soft_high].sum(axis=1)

        # Prevent division by zero
        soft_band = np.where(soft_band == 0, 1, soft_band)

        hardness_ratio = hard_band / soft_band
        intensity = hard_band + soft_band

        plt.figure(figsize=(8,6))
        plt.scatter(hardness_ratio, intensity, c='green', s=10, alpha=0.5)
        plt.title('Hardness-Intensity Diagram')
        plt.xlabel('Hardness Ratio (Hard/Soft)')
        plt.ylabel('Intensity (Hard + Soft)')
        plt.grid(True)
        plt.show()
        return hardness_ratio, intensity
    except KeyError as e:
        print(f"KeyError: {e}. Available columns: {data.columns.names}")
        raise
    except ValueError as ve:
        print(f"ValueError: {ve}")
        raise

# AI for Classification using KMeans
def cluster_hid(hardness_ratio, intensity):
    features = np.column_stack((hardness_ratio, intensity))
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(features)
    plt.figure(figsize=(8,6))
    plt.scatter(hardness_ratio, intensity, c=labels, cmap='viridis', s=10, alpha=0.5)
    plt.title('HID with Clusters')
    plt.xlabel('Hardness Ratio')
    plt.ylabel('Intensity')
    plt.grid(True)
    plt.show()

# Train Neural Network on Spectral Data
def train_nn(freqs, power):
    # Neural Network Model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(1,)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')

    # Prepare Data
    freqs_reshaped = freqs.reshape(-1, 1)  # (n_samples, 1)
    power_log = np.log10(power + 1e-10)  # Log-transform power to stabilize training

    # Train Model
    model.fit(freqs_reshaped, power_log, epochs=20, batch_size=32, verbose=1)

    # Predictions
    predictions_log = model.predict(freqs_reshaped)
    predictions = 10**predictions_log.flatten()

    # Plot Original vs Predicted
    plt.figure(figsize=(10,6))
    plt.loglog(freqs, power, label='Original', color='red')
    plt.loglog(freqs, predictions, label='Predicted', color='blue')
    plt.title('Power Density Spectrum: Original vs Predicted')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main Execution
if __name__ == "__main__":
    file_path = 'AS1G02_010T01_9000000252lxp1BB_level1.fits'  # Replace with your FITS file path
    data = load_fits(file_path)

    # Optional: Inspect DataArray contents
    print("\nInspecting the first few DataArray entries:")
    inspect_data_array(data, num_rows=5)

    # Generate Light Curve
    try:
        time, counts = generate_light_curve(data)
    except KeyError:
        print("Unable to generate light curve due to missing columns.")
    except Exception as e:
        print(f"Error generating light curve: {e}")

    # Generate Power Density Spectrum and Train Neural Network
    if 'time' in locals() and 'counts' in locals():
        if len(time) > 1:
            time_step = np.median(np.diff(time))
            freqs, power = generate_pds(counts, time_step)
            train_nn(freqs, power)
        else:
            print("Not enough time data points to compute power density spectrum.")

    # Generate Hardness-Intensity Diagram
    # Define energy bands based on channel indices
    # Example: hard band channels 100-500, soft band 50-100
    # User should adjust based on calibration
    energy_bands = [100, 500, 50, 100]  # Replace with appropriate channel indices
    try:
        hardness_ratio, intensity = generate_hid(data, energy_bands)
        # Cluster Hardness-Intensity Diagram
        cluster_hid(hardness_ratio, intensity)
    except KeyError:
        print("Unable to generate HID due to missing columns.")
    except ValueError as ve:
        print(f"ValueError: {ve}")
    except Exception as e:
        print(f"Error generating HID: {e}")