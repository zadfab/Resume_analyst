import streamlit as st
from rag_processing import load_and_split_store, get_chat_details, get_cover_letter, get_ats_detials

PAGES = {
    "home": "🏠 Home",
    "match": "🎯 Job Match",
    "cover": "✍️ Cover Letter",
}

# Current page from query params
current_page = st.query_params.get("page", "home")

# Navbar with buttons (updates URL without reload)
cols = st.columns(len(PAGES))
for i, (page, label) in enumerate(PAGES.items()):
    if cols[i].button(label):
        st.query_params.update({"page": page})
        current_page = page

def text_area_changed():
        st.session_state["stored_jd_text"] = st.session_state.get("jd_text_key","")



# Page content
if "resume_file" not in st.session_state:
    st.session_state.resume_file = None

if current_page == "home":
    st.title("🏠 Home")
    if st.session_state.resume_file:
        file = st.session_state.resume_file
        st.success(f"✅ File in session: {file.name} ({file.size} bytes)")

        if st.button("Load and Chat"):
            with st.spinner("Loading your resume. Please wait"):
                current_retriever = load_and_split_store(file)
                st.session_state.retriever_state = current_retriever
                st.balloons()
        if retriever:=st.session_state.get("retriever_state",None):
            user_query = st.text_area(label="Chat with your Resume")
            if st.button("Submit"):
                response = get_chat_details(retriever,user_query)
                st.write(response)
                
        if st.button("Upload a different file"):
            st.session_state.resume_file = None
            st.session_state.retriever_state = None
            st.rerun()
    else:
        uploaded = st.file_uploader("Upload your resume (PDF/DOCX)", type=["pdf", "docx"])
        if uploaded:
            st.session_state.resume_file = uploaded
            st.rerun()


elif current_page == "match":
    st.title("🎯 Job Match")

    jd_text = st.text_area(value = st.session_state.get("stored_jd_text",""),
        label="Add your Job description here for comparison",
        key="jd_text_key",
        on_change=text_area_changed
    )
    if st.button("Get ATS score"):
        if jd_text:=st.session_state.get("stored_jd_text",""):
            if st.session_state.get("resume_file") and st.session_state.get("retriever_state"):
                with st.spinner("Getting ATS score. Please wait"):
                    ats_obj = get_ats_detials(jd_text)
                    st.subheader("Your ATS score")
                    st.write(ats_obj.score)
                    st.subheader("Matching skills : ")
                    st.markdown("\n* ".join(ats_obj.matched_skills))
                    st.subheader("Missing skills : ")
                    st.markdown("\n* ".join(ats_obj.missing_skills))
            else:
                st.write("🙏🏻 please upload your resume in home page and load the document")
   
        else:
            st.toast("Add your Job Description 😞")
        
           
               
    


elif current_page == "cover":
    st.title("✍️ Cover Letter Generator")
    if st.session_state.get("resume_file") and st.session_state.get("retriever_state"):
        if jd := st.session_state.get("stored_jd_text",""):

            if st.button("Generate cover letter"):
                with st.spinner("Generating cover letter. Please wait"):
                    cover_letter_text, pdf_bytes = get_cover_letter(jd)
                    st.download_button(
                    label="📥 Download Cover Letter PDF",
                    data=pdf_bytes,
                    file_name="cover_letter.pdf",
                    mime="application/pdf"
                    )
                    st.subheader("Your cover letter content")
                    st.markdown(cover_letter_text)

        else:
            st.write("🙏🏻 Please add job description in job match page")
    else:
        st.write("🙏🏻 Please upload your resume in home page and load the document")