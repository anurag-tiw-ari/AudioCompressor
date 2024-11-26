import os
import numpy as np
import pywt
import scipy.io.wavfile as wav
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Set the folder to save the uploaded and compressed files
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to apply wavelet transform and compress audio
def compress_audio(input_path, output_path, threshold=0.1, wavelet='db1', level=4):
    # Read input audio
    fs, audio_data = wav.read(input_path)

    # Check if audio data is stereo (2 channels)
    if len(audio_data.shape) == 2:
        audio_data = audio_data[:, 0]  # Use only the first channel (mono)

    # Apply discrete wavelet transform (DWT) to the audio signal
    coeffs = pywt.wavedec(audio_data, wavelet, level=level)

    # Apply thresholding to wavelet coefficients
    coeffs_thresholded = [pywt.threshold(c, threshold * max(c)) for c in coeffs]

    # Reconstruct the audio signal from thresholded coefficients
    compressed_audio = pywt.waverec(coeffs_thresholded, wavelet)

    # Ensure the compressed audio has the same length as the original
    compressed_audio = compressed_audio[:len(audio_data)]

    # Normalize the output to the same range as the original
    compressed_audio = np.int16(compressed_audio / np.max(np.abs(compressed_audio)) * 32767)

    # Write the compressed audio to the output file
    wav.write(output_path, fs, compressed_audio)

# Route to handle the file upload and compression
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and file.filename.endswith('.wav'):
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Define the compressed file path
        compressed_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'compressed_' + file.filename)

        # Compress the audio
        compress_audio(file_path, compressed_file_path)

        return jsonify({
            'message': 'File successfully uploaded and compressed',
            'filename': 'compressed_' + file.filename
        }), 200

    else:
        return jsonify({'error': 'Invalid file type, only .wav files are allowed'}), 400

# Route to serve the compressed file for download
@app.route('/uploads/<filename>')
def download_file(filename):
    # Send the file for download
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)

