
from langchain_google_genai import GoogleGenerativeAI
from rank_bm25 import BM25Okapi


from langchain_core.documents import Document
from typing import List, Union
import re


import requests
import random 
import os
import json


import gspread
from oauth2client.service_account import ServiceAccountCredentials 
from langchain.memory import ConversationBufferMemory




#def build_bm25_corpus(pdf_path):
  #  """Extracts text, processes it, and creates a BM25 indexable corpus."""
    #raw_text = extract_text_from_pdf(pdf_path)
    #cleaned_text = clean_text(raw_text)
    #text_chunks = split_text(cleaned_text)
  #  chunker = PureSemanticChunker(api_key="") 
   # text=load_pdf(extracted_data) 
    #text_chunks = chunker.chunk(extracted_data)
    # Prepare BM25 corpus
    #bm25_corpus = [chunk for chunk in text_chunks if len(chunk) > 20]
    # Save as JSON (for debugging)
    #with open("bm25_corpus.json", "w", encoding="utf-8") as f:
	#json.dump(bm25_corpus, f, indent=4)
	    
    #return bm25_corpus


     

#Create text chunks
#def text_split(extracted_data):
	#chunker = PureSemanticChunker(api_key="") 
	#text=load_pdf(extracted_data) 
	#text_chunks = chunker.chunk(extracted_data)
	
   # text_splitter = RecursiveCharacterTextSplitter(chunk_size = 600, chunk_overlap =150)
    #text_chunks = text_splitter.split_documents(extracted_data)

    #return text_chunks
#download embedding model

##### otp handeling part####### 

# Twilio Credentials 
# TWILIO_ACCOUNT_SID = ""
# TWILIO_AUTH_TOKEN = ""
# TWILIO_PHONE_NUMBER = ""

# USER_DATA_FILE = "static/user_data.json"

# def generate_otp():
#     """Generate a 4-digit OTP."""
#     return str(random.randint(1000, 9999))

# def send_otp_sms(phone, otp):
#     """Send OTP using Twilio API with a valid Twilio number."""
#     if not phone.startswith("+"):
#         phone = "+91" + phone  # Ensure country code is added for India

#     client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

#     try:
#         message = client.messages.create(
#             body=f"Your IEM Chatbot OTP is: {otp}",
#             from_=TWILIO_PHONE_NUMBER,  # Must be Twilio's SMS-enabled number
#             to=phone
#         )
#         print(f"OTP sent to {phone}. Twilio SID: {message.sid}")
#     except Exception as e:
#         print(f"Twilio Error: {e}")

# def save_user_phone(phone):
#     """Store user phone number in a file only if OTP is verified."""
#     # Ensure the file exists and is properly initialized
#     if not os.path.exists(USER_DATA_FILE) or os.stat(USER_DATA_FILE).st_size == 0:
#         with open(USER_DATA_FILE, "w") as file:
#             json.dump([], file)  # Initialize with an empty list

#     # Load existing data safely
#     try:
#         with open(USER_DATA_FILE, "r") as file:
#             data = json.load(file)  # Read JSON content
#     except json.JSONDecodeError:  
#         data = []  # Reset to an empty list if JSON is corrupted

#     # Avoid duplicate phone numbers
#     if phone not in [entry["phone"] for entry in data]:
#         data.append({"phone": phone})

#     # Save updated data
#     with open(USER_DATA_FILE, "w") as file:
#         json.dump(data, file, indent=4)

###### phone number store part ########

# USER_DATA_FILE = "static/user_data.json"

# Ensure user data file exists
# if not os.path.exists(USER_DATA_FILE):
#     with open(USER_DATA_FILE, "w") as file:
#         json.dump([], file)

# # Function to save phone number
# def save_user_phone(phone):
#     with open(USER_DATA_FILE, "r+") as file:
#         try:
#             data = json.load(file)
#         except json.JSONDecodeError:
#             data = []  # Initialize if empty

#         if phone not in data:
#             data.append(phone)
#             file.seek(0)
#             json.dump(data, file)
#             file.truncate()  

#### store phone number inside google sheet ##### 

# Set up Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = ServiceAccountCredentials.from_json_keyfile_name("static/iemphone-6fec50f7ae8a.json", scope)
client = gspread.authorize(creds)

# Open the Google Sheet (Replace 'iem' with your sheet name)
SHEET_NAME = "iembotdata"
sheet = client.open(SHEET_NAME).sheet1  # Select the first sheet

def save_user_phone(phone):
    """Save phone number to Google Sheet."""
    try:
        existing_numbers = sheet.col_values(1)  # Get all phone numbers from the first column

        if phone not in existing_numbers:
            sheet.append_row([phone])  # Append phone number only if it doesn't exist
            print(f"Phone number {phone} saved to Google Sheets.")
        else:
            print(f"Phone number {phone} already exists in Google Sheets.")
    except Exception as e:
        print(f"Error saving phone number: {e}") 


#### memmory mentain ##### 

