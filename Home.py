import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import nest_asyncio
from llama_parse import LlamaParse
import re
import tempfile
import boto3
import uuid
from langchain.document_loaders import AmazonTextractPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI




# Apply nest_asyncio
nest_asyncio.apply()



# Set up LlamaParse
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# Set up LlamaParse
llama_parser = LlamaParse(api_key=st.secrets["LLAMA_CLOUD_API_KEY"], result_type="text")

# AWS clients
s3_client = boto3.client('s3',
                         region_name=st.secrets["AWS_REGION"],
                         aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
                         aws_secret_access_key=st.secrets["AWS_SECRET_KEY"])

textract_client = boto3.client(
    "textract",
    region_name=st.secrets["AWS_REGION"],
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY"],
    aws_secret_access_key=st.secrets["AWS_SECRET_KEY"]
)


# Alert System functions
def process_excel_file(uploaded_file):
    tracker_df = pd.read_excel(uploaded_file, sheet_name='AuthorizationReferral Tracker')
    
    tracker_df['Termed Date'] = pd.to_datetime(tracker_df['Termed Date'], errors='coerce')
    
    current_date = datetime.now()
    
    def alert_reason(row):
        reasons = []
        if row['Termed Date'] <= current_date + timedelta(days=7):
            if row['Termed Date'] < current_date:
                reasons.append("Termed date has passed")
            else:
                reasons.append("Termed date is near")
        if row['Visit Remaining'] < 3:
            reasons.append("Visits remaining are less than 3")
        
        return ", ".join(reasons)
    
    tracker_df['Alert Reason'] = tracker_df.apply(alert_reason, axis=1)
    return tracker_df

def display_alerts(tracker_df):
    alert_df = tracker_df[tracker_df['Alert Reason'] != ""]
    
    if not alert_df.empty:
        st.subheader("Patient Alerts")
        for _, row in alert_df.iterrows():
            with st.expander(f"Alert for {row['Patient Name']}"):
                st.write(f"**Patient Name:** {row['Patient Name']}")
                st.write(f"**Termed Date:** {row['Termed Date'].strftime('%Y-%m-%d')}")
                st.write(f"**Visits Remaining:** {row['Visit Remaining']}")
                st.write(f"**Alert Reason:** {row['Alert Reason']}")
    else:
        st.success("No alerts triggered at the moment.")

# EOB Processor functions
def extract_payment_info(text, paid_amount):
    if float(paid_amount) > 0:
        eft_pattern = r"EFT\s*NUMBER[:\s]*([A-Z0-9\-]+)[\s\S]*?EFT\s*DATE[:\s]*([0-9/]+)[\s\S]*?EFT\s*AMOUNT[:\s]*\$?([0-9,]+\.[0-9]{2})"
        match = re.search(eft_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            eft_number = match.group(1).strip()
            eft_date = match.group(2).strip()
            eft_amount = match.group(3).strip().replace(',', '')  # Remove commas for large numbers
            return eft_number, eft_date, eft_amount
        else:
            return None, None, None
    else:
        return "N/A", "N/A", "N/A"

def extract_claim_number(text):
    pattern = r"Claim Number\s*[\s\S]*?(\d+)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def extract_grand_totals(text):
    pattern = r'Grand Totals:\s*Other Patient\s*Line Charge\s*Allowed\s*QPA\s*Contractual\s*Payer Initiated\s*OA\s*Copay\s*Deductible\s*Coinsurance\s*Responsibility\s*Withhold\s*Paid\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})\s*(\$\d+\.\d{2})'
    match = re.search(pattern, text)

    categories = ["Line Charge", "Allowed", "QPA", "Contractual", "Payer Initiated", "OA", "Copay",
                  "Deductible", "Coinsurance", "Responsibility", "Withhold", "Paid"]

    if match:
        return dict(zip(categories, match.groups()))
    else:
        return dict(zip(categories, ["$0.00"] * 12))  # Return zeros if no match

def extract_corrected_patient_name(text):
    pattern = r"Corrected Patient Name:\s+([A-Za-z,\s]+)"
    matches = re.findall(pattern, text)
    return matches[0] if matches else None

def extract_date_of_service(text):
    date_pattern = r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})-'
    matches = re.findall(date_pattern, text)
    return matches[0] if matches else (None, None)

def extract_pdf_data(pdf_path):
    document = llama_parser.load_data(pdf_path)

    if len(document) > 0:
        text = document[0].text
    else:
        st.warning(f"Skipping: No content found in {pdf_path}")
        return None

    claim_number = extract_claim_number(text)
    grand_totals = extract_grand_totals(text)
    start_date, end_date = extract_date_of_service(text)

    grand_totals_mapped = {key: float(value.strip('$')) for key, value in grand_totals.items()}
    total_ptr = grand_totals_mapped["Copay"] + grand_totals_mapped["Coinsurance"] + grand_totals_mapped["Deductible"]
    check_eft_number, eft_date, eft_amount = extract_payment_info(text, grand_totals_mapped["Paid"])
    claim_number_str = str(claim_number) if claim_number else "N/A"
    corrected_patient_name = extract_corrected_patient_name(text)

    data = {
        'Patient Name': corrected_patient_name,
        'Date of Service': start_date if start_date else "N/A",
        'Line Charge': grand_totals_mapped["Line Charge"],
        'Allowed': grand_totals_mapped["Allowed"],
        'Contractual': grand_totals_mapped["Contractual"],
        'Copay': grand_totals_mapped["Copay"],
        'Deductible': grand_totals_mapped["Deductible"],
        'Coinsurance': grand_totals_mapped["Coinsurance"],
        'Paid': grand_totals_mapped["Paid"],
        'Total PTR': total_ptr,
        'Check/EFT Number': check_eft_number,
        'EFT Amount': eft_amount,
        'EFT Date': eft_date,
        'Processed/Denial Date': eft_date,
        'Payer Claim Number': claim_number_str
    }

    return data

def process_all_pdfs(pdf_files):
    all_data = []

    for uploaded_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        pdf_data = extract_pdf_data(temp_file_path)
        if pdf_data:
            all_data.append(pdf_data)
        else:
            st.warning(f"Skipping: No data extracted from {uploaded_file.name}")

        os.unlink(temp_file_path)

    if all_data:
        df = pd.DataFrame(all_data)

        df['Date of Service'] = pd.to_datetime(df['Date of Service'], errors='coerce', format='%m/%d/%Y')
        df['Date of Service'] = df['Date of Service'].dt.date

        df_sorted = df.sort_values(by='Date of Service')

        df_sorted['Payer Claim Number'] = df_sorted['Payer Claim Number'].apply(lambda x: f'{int(x):d}' if pd.notnull(x) and x != "N/A" else x)

        return df_sorted
    else:
        st.error("No valid data extracted from PDFs.")
        return None

# Referral Verification functions
def upload_to_s3(local_file_path):
    try:
        unique_filename = f"{uuid.uuid4().hex}_{os.path.basename(local_file_path)}"
        s3_file_path = f"pdf_files/{unique_filename}"
        
        bucket_name = st.secrets["BUCKET_NAME"]
        s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
        return f"s3://{bucket_name}/{s3_file_path}"
    except Exception as e:
        st.error(f"Error uploading to S3: {str(e)}")
        return None

def process_with_textract(s3_uri):
    try:
        bucket_name, s3_key = s3_uri.replace("s3://", "").split("/", 1)
        
        response = textract_client.start_document_text_detection(
            DocumentLocation={'S3Object': {'Bucket': bucket_name, 'Name': s3_key}}
        )
        job_id = response['JobId']
        
        st.write("Processing document with Textract...")
        progress_bar = st.progress(0)
        
        while True:
            response = textract_client.get_document_text_detection(JobId=job_id)
            status = response['JobStatus']
            progress_bar.progress(0.5)
            
            if status in ['SUCCEEDED', 'FAILED']:
                break
        
        progress_bar.progress(1.0)
        
        if status == 'SUCCEEDED':
            pages = []
            pages.append(response)
            
            while 'NextToken' in response:
                response = textract_client.get_document_text_detection(
                    JobId=job_id,
                    NextToken=response['NextToken']
                )
                pages.append(response)
            
            text = ""
            for page in pages:
                for item in page['Blocks']:
                    if item['BlockType'] == 'LINE':
                        text += item['Text'] + "\n"
            
            return text
        else:
            st.error("Textract processing failed")
            return None
            
    except Exception as e:
        st.error(f"Error processing with Textract: {str(e)}")
        return None

def process_structured_pdf(pdf_path):
    try:
        parsed_content = llama_parser.load_data(pdf_path)
        return parsed_content.text if hasattr(parsed_content, 'text') else str(parsed_content)
    except Exception as e:
        st.error(f"Error processing structured PDF: {str(e)}")
        return None

def process_and_query(text, query):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=32,
            length_function=len,
        )
        texts = text_splitter.split_text(text)

        embeddings = OpenAIEmbeddings()
        docsearch = FAISS.from_texts(texts, embeddings)

        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        docs = docsearch.similarity_search(query)
        return chain.run(input_documents=docs, question=query)
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def process_pdf(pdf_path, is_scanned):
    try:
        if is_scanned:
            s3_uri = upload_to_s3(pdf_path)
            if not s3_uri:
                return None
            
            text = process_with_textract(s3_uri)
        else:
            text = process_structured_pdf(pdf_path)
        
        if not text:
            return None

        query = """Extract the following details from the pdf:
            1. Member Name
            2. Member ID
            3. PCP/Provider Name
            4. PCP/Provider NPI
            5. Specialist Organization Name
            6. Specialist NPI
            7. Diagnosis Code
            And tell me what all information is missing. Also summarize the whole information."""
        
        return process_and_query(text, query)
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Main application
def main():
    st.set_page_config(page_title="Healthcare Services Dashboard", page_icon="🏥", layout="wide")
    
    # Top navigation
    st.markdown('<div style="text-align: center;">', unsafe_allow_html=True)
    page = st.radio("", ("Home", "Services", "About Us"), horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if page == "Home":
        home()
    elif page == "Services":
        services()
    elif page == "About Us":
        about_us()

def home():
    # Add logo
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("macksofy_white.png", use_column_width=True)
    
    st.title("Welcome to Healthcare Services Dashboard")
    st.write("Streamlining healthcare processes with cutting-edge technology.")

    service_col1, service_col2, service_col3 = st.columns(3)
    with service_col1:
        st.subheader("Patient Alert System")
        st.image("alert2.png", use_column_width=True)
        st.write("Monitor patient status and generate timely alerts.")
    
    with service_col2:
        st.subheader("PDF-EOB Processor")
        st.image("eob.png", use_column_width=True)
        st.write("Extract and process data from EOB PDFs efficiently.")
    
    with service_col3:
        st.subheader("Referral Verification")
        st.image("referral.png", use_column_width=True)
        st.write("Verify Patient Referral Data.")

def services():
    st.title("Our Services")
    service = st.selectbox("Choose a service", ["Patient Alert System", "PDF-EOB Processor", "Referral Verification"])

    if service == "Patient Alert System":
        st.subheader("Patient Alert System")
        uploaded_file = st.file_uploader("Upload Excel file", type="xlsx")
        if uploaded_file is not None:
            tracker_df = process_excel_file(uploaded_file)
            display_alerts(tracker_df)

    elif service == "PDF-EOB Processor":
        st.subheader("PDF-EOB Processor")
        pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if pdf_files:
            processed_df = process_all_pdfs(pdf_files)
            if processed_df is not None:
                st.dataframe(processed_df)
                csv = processed_df.to_csv(index=False)
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="processed_eob_data.csv",
                    mime="text/csv",
                )

    elif service == "Referral Verification":
        st.subheader("Referral Verification")
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
        is_scanned = st.checkbox("Is this a scanned PDF?")
        
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            
            result = process_pdf(temp_file_path, is_scanned)
            if result:
                st.write(result)
            else:
                st.error("Failed to process the PDF.")
            
            os.unlink(temp_file_path)

def about_us():
    st.title("About Us")
    st.write("""
    We are a leading healthcare technology company dedicated to improving patient care and streamlining healthcare processes. 
    Our suite of services is designed to enhance efficiency, reduce errors, and provide timely insights to healthcare providers. 
    Our mission is to revolutionize healthcare management through innovative technology solutions, ensuring better outcomes for patients and providers alike.
    """)
    st.image("macksofy_white.png", use_column_width=True)

if __name__ == "__main__":
    main()