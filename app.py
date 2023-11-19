from flask import Flask, request, jsonify
from process import predict_category_from_pdf

app = Flask(__name__)

@app.route('/predict_pdf', methods=['POST'])
def predict_pdf():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    pdf_file = request.files['file']
    if pdf_file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save uploaded file
    pdf_path = 'uploaded_cv.pdf'
    pdf_file.save(pdf_path)

    # Predict category from PDF
    predicted_category = predict_category_from_pdf(pdf_path)

    return jsonify({'predicted_category': predicted_category})

if __name__ == '__main__':
    app.run(debug=True)
