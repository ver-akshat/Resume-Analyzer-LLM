import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
import warnings
warnings.filterwarnings('ignore')

def streamlit_config():
    # set page configuration
    st.set_page_config(page_title='Resume Analyzer AI', layout="wide")
    # page header transparent color
    page_background_color = """
    <style>

    [data-testid="stHeader"] 
    {
    background: rgba(10,10,10,10);
    }
    </style>
    """
    st.markdown(page_background_color,unsafe_allow_html=True)
    # title and position
    st.markdown(f'<h1 style="text-align: center;">AI-Powered Resume Analyzer</h1>',unsafe_allow_html=True)


class ResumeAnalyzer:

    def pdf_to_chunks(pdf):
        # read pdf and it returns memory address
        pdf_reader=PdfReader(pdf)
        # extrat text from each page separately
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        # Split the long text into small chunks.
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=200,
            length_function=len)
        chunks = text_splitter.split_text(text=text)
        return chunks


    def resume_summary(query_with_chunks):
        query=f''' need to detailed summarization of below resume and finally conclude them

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_strength(query_with_chunks):
        query=f'''need to detailed analysis and explain of the strength of below resume and finally conclude them
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def resume_weakness(query_with_chunks):
        query=f'''need to detailed analysis and explain of the weakness of below resume and how to improve make a better resume.

                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def job_title_suggestion(query_with_chunks):

        query=f''' what are the job roles the candidate can apply to likedin based on below,along with expected pay-scale in indian rupees?
                    
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    {query_with_chunks}
                    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                    '''
        return query


    def openai(openai_api_key, chunks, analyze):
        # Using OpenAI service for embedding
        embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)
        # Facebook AI Similarity Serach library help us to convert text data to numerical vector
        vectorstores=FAISS.from_texts(chunks,embedding=embeddings)
        # compares the query and chunks, enabling the selection of the top 'K' most similar chunks based on their similarity scores.
        docs=vectorstores.similarity_search(query=analyze,k=3)
        # creates an OpenAI object, using the ChatGPT 3.5 Turbo model
        llm=ChatOpenAI(model='gpt-3.5-turbo',api_key=openai_api_key)
        # question-answering (QA) pipeline, making use of the load_qa_chain function
        chain=load_qa_chain(llm=llm,chain_type='stuff')
        response=chain.run(input_documents=docs,question=analyze)
        return response

streamlit_config()
add_vertical_space(1)


# sidebar
with st.sidebar:
    add_vertical_space(3)
    option = option_menu(menu_title='',options=['Summary','Strength','Weakness','Job Titles','Exit'],
                         icons=['house-fill','database-fill','pass-fill','list-ul','sign-turn-right-fill'])

if option == 'Summary':

    # file upload
    pdf=st.file_uploader(label='',type='pdf')
    openai_api_key = st.text_input(label='OpenAI API Key',type='password')

    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks=ResumeAnalyzer.pdf_to_chunks(pdf)
            summary=ResumeAnalyzer.resume_summary(query_with_chunks=pdf_chunks)
            result_summary=ResumeAnalyzer.openai(openai_api_key=openai_api_key, chunks=pdf_chunks, analyze=summary)
            st.subheader('Summary:')
            st.write(result_summary)
    except Exception as e:
        col1,col2=st.columns(2)
        with col1:
            st.warning(e)

elif option == 'Strength':
    # file upload
    pdf=st.file_uploader(label='',type='pdf')
    openai_api_key=st.text_input(label='OpenAI API Key',type='password')
    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks=ResumeAnalyzer.pdf_to_chunks(pdf)
            # Resume summary
            summary=ResumeAnalyzer.resume_summary(query_with_chunks=pdf_chunks)
            result_summary=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=summary)
            strength=ResumeAnalyzer.resume_strength(query_with_chunks=result_summary)
            result_strength=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=strength)
            st.subheader('Strength:')
            st.write(result_strength)

    except Exception as e:
        col1,col2 =st.columns(2)
        with col1:
            st.warning(e)


elif option == 'Weakness':
    # file upload
    pdf=st.file_uploader(label='',type='pdf')
    openai_api_key=st.text_input(label='OpenAI API Key',type='password')
    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks=ResumeAnalyzer.pdf_to_chunks(pdf)
            # Resume summary
            summary=ResumeAnalyzer.resume_summary(query_with_chunks=pdf_chunks)
            result_summary=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=summary)
            weakness=ResumeAnalyzer.resume_weakness(query_with_chunks=result_summary)
            result_weakness=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=weakness)
            st.subheader('Weakness:')
            st.write(result_weakness)

    except Exception as e:
        col1,col2=st.columns(2)
        with col1:
            st.warning(e)


elif option == 'Job Titles':
    # file upload
    pdf=st.file_uploader(label='',type='pdf')
    openai_api_key=st.text_input(label='OpenAI API Key',type='password')

    try:
        if pdf is not None and openai_api_key is not None:
            pdf_chunks=ResumeAnalyzer.pdf_to_chunks(pdf)
            # Resume summary
            summary=ResumeAnalyzer.resume_summary(query_with_chunks=pdf_chunks)
            result_summary=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=summary)
            job_suggestion=ResumeAnalyzer.job_title_suggestion(query_with_chunks=result_summary)
            result_suggestion=ResumeAnalyzer.openai(openai_api_key=openai_api_key,chunks=pdf_chunks,analyze=job_suggestion)
            st.subheader('Suggestion: ')
            st.write(result_suggestion)
    except Exception as e:
        col1, col2 = st.columns(2)
        with col1:
            st.warning(e)
    except:
        st.write('')
        st.info("This feature is currently not working in the deployed Streamlit application due to a 'selenium.common.exceptions.WebDriverException' error.")
        st.write('')
        st.write("Please use the local Streamlit application for a smooth experience: [http://localhost:8501](http://localhost:8501)")

elif option == 'Exit':
    add_vertical_space(3)
    col1,col2,col3=st.columns([0.3,0.4,0.3])
    with col2:
        st.success('Thank you for your time. Exiting the application')
        st.balloons()

