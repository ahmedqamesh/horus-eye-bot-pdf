import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import worker  # Import the worker module

# Initialize Flask app and CORS
app = Flask(__name__)

# Ensure the database directory exists
UPLOAD_DIR = os.path.join(os.getcwd(), "database")
os.makedirs(UPLOAD_DIR, exist_ok=True)


cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json.get('userMessage', '')  # Extract the user's message
    try:
        bot_response = worker.process_prompt(user_message)  # Process the user's message
    except Exception as e:
        return jsonify({"botResponse": f"Error: {str(e)}"}), 500

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try "
                           "again. If the problem persists, try using a different file"
        }), 400

    file = request.files['file']  # Extract the uploaded file

    # Save the uploaded file in /database/
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(file_path)

    try:
        worker.process_document(file_path)  # Process the document
    except Exception as e:
        return jsonify({
            "botResponse": f"Failed to process the document: {str(e)}"
        }), 500

    # Return the success message
    return jsonify({
        "botResponse": "Analyzing PDF..., Please ask the question!"
    }), 200

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Run the Flask app
if __name__ == "__main__":
    app.logger.info("INFO:Flask app running at http://localhost:8000")
    app.run(debug=True, port=8000, host='0.0.0.0')
