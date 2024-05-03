import streamlit as st

def show_pdf(pdf_url):
    st.markdown(
        f"<iframe src='{pdf_url}' width='130%' height='700' style='border:none; margin-left: -150px;'></iframe>",
        unsafe_allow_html=True
    )

def main():
    st.markdown("""
        <style>
        .title {
            margin-left: -150px;
            margin-right: -100px;
            overflow-wrap: normal;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="title">PWC Leases Guide</h1>', unsafe_allow_html=True)
    
    # st.title('PDF Viewer for Document 1')
    pdf_url = "https://drive.google.com/file/d/16g6oGhEfbzweLylv123-vpfVEtJB2Oy7/preview"  # Update this to your actual PDF URL
    show_pdf(pdf_url)

if __name__ == "__main__":
    main()

