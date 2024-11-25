import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_google_genai import GoogleGenerativeAI
from langchain.chains import RetrievalQA

st.header('Introduce 📃')
st.markdown("File Q&A là một hệ thống tự động hóa giúp bạn nhanh chóng truy xuất thông tin từ các tài liệu chẳng hạn như file PDF. Hệ thống được thiết kế để hỗ trợ trả lời câu hỏi từ nội dung tài liệu một cách chính xác và hiệu quả. 🦾")
with st.sidebar:
    API_input = st.text_input("Gemini API Key" , type="password")
    st.title('Your Document')
    file = st.file_uploader("Upload a file PDF file and start asking question", type ='pdf')

if file is not None:
    pdf_reader = PdfReader(file)
    text= ""
    for page in pdf_reader.pages:
        text+= page.extract_text()

    # Cơ chế chunk nha mấy ní
    text_splitter  = RecursiveCharacterTextSplitter(
        separators= '\n', # Ra yêu cầu phân tách các đoạn bằng kí tự xuống dòng
        chunk_size = 1000, # độ dài của mỗi đoạn là 1000 kí tự
        chunk_overlap = 150, # mang 150 kí tự từ đoạn trước vào đoạn sau để tạo mối liên kết giữa các đoạn
        length_function = len
    )

    chunk = text_splitter.split_text(text)
    chunk = [text.replace('\n', '') for text in chunk]
    user_input = st.text_input("Enter what you want to ask the file here 👇" , key="user_input")

    if user_input:
        if not API_input:
            st.warning("Please enter your [API](https://ai.google.dev/) into sidebar.")
        else:
            try:
                embeddings = GoogleGenerativeAIEmbeddings(google_api_key=API_input, model="models/embedding-001")

                vectorstore = DocArrayInMemorySearch.from_texts(chunk,
                    embedding=embeddings 
                )

                llm = GoogleGenerativeAI(
                        model="gemini-pro",
                        temperature=0.7,
                        google_api_key=API_input
                    )

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever()
                )

                answer = qa_chain.invoke(user_input)
                st.write(answer['result'])
            except Exception:
                st.error('Please enter the correct [API](https://ai.google.dev/) Key')

else:
    st.info('Please upload a file')