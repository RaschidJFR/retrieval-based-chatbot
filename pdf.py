import PyPDF2

def get_documentation_text():
  return extract_text_from_pdf('sample.pdf')

def extract_text_from_pdf(pdf_path):
    # Open the PDF file in read-binary mode
    with open(pdf_path, 'rb') as file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        # Iterate through each page in the PDF
        for page in pdf_reader.pages:
            # Extract text from the page and add it to the text variable
            text += page.extract_text() + ' '  # Adding a space for separation between pages
    return text
