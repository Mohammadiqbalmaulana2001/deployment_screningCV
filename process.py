from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle
import numpy as np
import PyPDF2

# Fungsi untuk mengekstrak teks dari dokumen PDF
def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def predict_category_from_pdf(pdf_path): 
    # Load model dan file pickle
    model = load_model('model/ScreningCV.h5')

    with open('model/le.pkl', 'rb') as le_file:
        label_encoder = pickle.load(le_file)

    with open('model/tokenizers.pkl', 'rb') as tok_file:
        tokenizer = pickle.load(tok_file)

    max_seq_length = 100

    # Extract text from PDF
    cv_text = extract_text_from_pdf(pdf_path)

    # Tokenize and predict
    new_cv_tokens = tokenizer.texts_to_sequences([cv_text])

    # Pad sequences to a fixed length
    new_cv_tokens_padded = pad_sequences(new_cv_tokens, maxlen=max_seq_length, padding='post')

    predicted_probabilities = model.predict(new_cv_tokens_padded)
    predicted_label = np.argmax(predicted_probabilities, axis=1)[0]
    predicted_category = label_encoder.inverse_transform([predicted_label])[0]

    return predicted_category

