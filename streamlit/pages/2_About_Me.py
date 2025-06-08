import streamlit as st

st.set_page_config(page_title=None,
                   page_icon=":shark:",
                   layout="wide",
                   initial_sidebar_state="auto",
                   menu_items=None)

def show_pdf(pdf_url):
    st.markdown(
        f"<iframe src='{pdf_url}' width='90%' height='700' style='border:none; margin-left: -5px;'></iframe>",
        unsafe_allow_html=True
    )

def main():
    st.markdown("""
        <style>
        .title {
            margin-left: -10px;
            margin-right: -100px;
            overflow-wrap: normal;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="title">Created by Meredith McCanse</h1>', unsafe_allow_html=True)
    st.markdown("GitHub documentation can be found here: [README](https://github.com/mmccanse/accounting-ai/blob/main/README.md)", unsafe_allow_html=True)
    st.markdown("Main app code can be found here: [Video_Assistant.py](https://github.com/mmccanse/accounting-ai/blob/main/Video_Assistant.py) or through GitHub link at top right:", unsafe_allow_html=True)
    st.markdown("Connect with me on [LinkedIn](https://www.linkedin.com/in/meredithmccanse/)", unsafe_allow_html=True)
    st.divider()
    st.write("""I am a passionate and energetic CPA with 8 years of accounting experience and many years of prior business, 
             freelance, and life experience. I just completed a 6-month Artificial Intelligence bootcamp through 
             the University of Denver. I'm confident that AI will drastically change the accounting landscape and am 
             excited to be part of this shift. My sweet spot is thinking through process improvements and using 
             technology to build automation and streamline processes. I balance big picture 
             perspectives and detail orientation to engage in team collaborations as well as the detail work of day-to-day 
             operational accounting. CPA, MBA, Certificate from DU in AI, Alteryx Designer Core certified, and open to work!""")
    
    # st.title('PDF Viewer for Document 1')
    pdf_url = "https://drive.google.com/file/d/13uquN_k279Jx_zUarWtAPaG3pR_aw8Xp/preview" # Update this to your actual PDF URL
    show_pdf(pdf_url)

if __name__ == "__main__":
    main()
