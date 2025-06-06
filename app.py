from flask import Flask, request, jsonify, render_template, Response
import os
from werkzeug.utils import secure_filename
import create_predict_data
import predict
import json
import time
import tempfile

app = Flask(__name__)

# Use system temp directory for uploads in production
if os.environ.get('RAILWAY_ENVIRONMENT'):
    app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
    app.config['PROCESS'] = tempfile.gettempdir()
else:
    app.config['UPLOAD_FOLDER'] = 'uploads'
    app.config['PROCESS'] = 'process'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESS'], exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 100MB max file size
app.config['MAX_CONTENT_PATH'] = 255  # Maximum length of file path


def send_estimate_time(estimate_time):
    return f"data: {json.dumps({'type': 'estimate', 'estimate_time': estimate_time})}\n\n"


def send_result(result, success):
    return f"data: {json.dumps({'type': 'result','success':success, 'result': result})}\n\n"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload_file', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return Response(send_result({
                'success': False,
                'message': 'No file part'
            }),
                            mimetype='text/event-stream')

        file = request.files['file']
        if file.filename == '':
            return Response(send_result({
                'success': False,
                'message': 'No selected file'
            }),
                            mimetype='text/event-stream')

        # Check file extension
        allowed_extensions = {'mp3', 'wav'}
        if not '.' in file.filename or file.filename.rsplit(
                '.', 1)[1].lower() not in allowed_extensions:
            return Response(send_result({
                'success':
                False,
                'message':
                'Invalid file type. Only MP3 and WAV files are allowed.'
            }),
                            mimetype='text/event-stream')

        # Check file size
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            return Response(send_result({
                'success':
                False,
                'message':
                f'File too large. Maximum size is {app.config["MAX_CONTENT_LENGTH"] // (1024*1024)}MB'
            }),
                            mimetype='text/event-stream')

        filename = secure_filename(file.filename)
        input_filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        output_filepath = os.path.join(app.config['PROCESS'])
        # Check if file path is too long
        if len(input_filepath) > app.config['MAX_CONTENT_PATH']:
            return Response(send_result({
                'success': False,
                'message': 'File path too long'
            }),
                            mimetype='text/event-stream')

        file.save(input_filepath)

        def generate():
            try:
                estimate_time = 2 * (
                    create_predict_data.split_number(input_filepath) -
                    1) if create_predict_data.split_number(
                        input_filepath) >= 2 else 3
                yield send_estimate_time(estimate_time)
                # Process audio file
                features_df = create_predict_data.process_audio_files(
                    input_filepath, output_filepath)

                result = predict.predict_adhd(features_df)
                print(result)

                yield send_result(result, True)

            except Exception as e:
                yield send_result({
                    'success': False,
                    'message': f'Error processing file: {str(e)}'
                })
            finally:
                # Clean up
                if os.path.exists(input_filepath):
                    os.remove(input_filepath)

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return Response(send_result({
            'success': False,
            'message': f'Server error: {str(e)}'
        }),
                        mimetype='text/event-stream')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
