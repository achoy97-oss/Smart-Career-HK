import streamlit as st
import sqlite3
import pinecone
import pandas as pd

from backend import JobSeekerBackend
from backend import LinkedInJobSearcher
from backend import get_all_jobs_for_matching
from backend import get_all_job_seekers
from backend import analyze_match_simple
from backend import show_match_statistics
from backend import show_instructions

from backend import get_jobs_for_interview
from backend import get_job_seeker_profile
from backend import ai_interview_page

from database import JobSeekerDB
from database import HeadhunterDB

db = JobSeekerDB()
db2 = HeadhunterDB()

from database import save_job_seeker_info
from database import save_head_hunter_job
from database import init_database
from database import init_head_hunter_database
from database import get_job_seeker_search_fields
from config import Config

import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Smart Career",
    page_icon="🎯",
    layout="wide"
)

# Initialize backend
@st.cache_resource
def load_backend():
    return JobSeekerBackend()

backend = load_backend()


# Initialize database
init_database()
init_head_hunter_database()

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "main"

# APP UI
def main_analyzer_page():
    """Main Page - Smart Career"""
    st.title("🎯 Smart Career")
    st.markdown("Upload your CV and let **GPT-4** find matching jobs globally, ranked by match quality!")

    # Define helper functions
    def smart_select_match(value, options):
        """Smart match select box options"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        for i, option in enumerate(options):
            if option.lower() in value_str or value_str in option.lower():
                return i
        return 0

    def format_ai_data(data, default=""):
        """Format AI returned data"""
        if isinstance(data, list):
            return ", ".join(data)
        elif isinstance(data, str):
            return data
        else:
            return default


    # Main Page - CV Upload Section
    st.header("📁 Upload Your CV")
    cv_file = st.file_uploader("Choose your CV", type=['pdf', 'docx'], key="cv_uploader")

    # Initialize variables
    autofill_data = {}
    analysis_complete = False
    ai_analysis = {}  # Initialize ai_analysis

    if cv_file:
        st.success(f"✅ Uploaded: **{cv_file.name}**")

        if st.button("🔍 Analyze with GPT-4", type="primary", use_container_width=True, key="analyze_button"):

            # STEP 1: Analyze Resume
            with st.spinner("🤖 Step 1/2: Analyzing your resume with GPT-4..."):
                try:
                    resume_data, ai_analysis = backend.process_resume(cv_file, cv_file.name)
                    
                    st.balloons()

                    # Display analysis results
                    st.markdown("---")
                    st.subheader("🤖 GPT-4 Career Analysis")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        primary_role = ai_analysis.get('primary_role', 'N/A')
                        st.metric("🎯 Primary Role", primary_role)

                    with col2:
                        confidence = ai_analysis.get('confidence', 0) * 100
                        st.metric("💯 Confidence", f"{confidence:.0f}%")

                    with col3:
                        st.metric("📊 Seniority", ai_analysis.get('seniority_level', 'N/A'))

                    # Skills detected by GPT-4
                    st.markdown("### 💡 Skills Detected by GPT-4")
                    skills = ai_analysis.get('skills', [])
                    if skills:
                        # Create skill tags
                        skills_html = ""
                        for skill in skills[:10]:
                            skills_html += f'<span style="background-color: #E8F4FD; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                        st.markdown(skills_html, unsafe_allow_html=True)

                        if len(skills) > 10:
                            with st.expander(f"➕ Show all {len(skills)} skills"):
                                more_skills_html = ""
                                for skill in skills[10:]:
                                    more_skills_html += f'<span style="background-color: #F0F0F0; padding: 5px 10px; margin: 3px; border-radius: 5px; display: inline-block;">{skill}</span> '
                                st.markdown(more_skills_html, unsafe_allow_html=True)
                    else:
                        st.warning("⚠️ No skills detected")

                    # Core strengths
                    st.markdown("### 💪 Core Strengths")
                    strengths = ai_analysis.get('core_strengths', [])
                    if strengths:
                        cols = st.columns(min(3, len(strengths)))
                        for i, strength in enumerate(strengths):
                            with cols[i % len(cols)]:
                                st.info(f"✓ {strength}")

                    # Extract and format data
                    autofill_data = {
                        # Educational background
                        "education_level": format_ai_data(ai_analysis.get('education_level', '')),
                        "major": format_ai_data(ai_analysis.get('major', '')),
                        "graduation_status": format_ai_data(ai_analysis.get('graduation_status', '')),
                        "university_background": format_ai_data(ai_analysis.get('university_background', '')),
                        
                        # Languages and certificates
                        "languages": format_ai_data(ai_analysis.get('languages', '')),
                        "certificates": format_ai_data(ai_analysis.get('certificates', '')),
                        
                        # Skills - directly use detected skills
                        "hard_skills": format_ai_data(skills),  # Use detected skills
                        "soft_skills": format_ai_data(ai_analysis.get('core_strengths', [])),  # Use core strengths
                        
                        # Work experience
                        "work_experience": format_ai_data(ai_analysis.get('work_experience', '')),
                        "project_experience": format_ai_data(ai_analysis.get('project_experience', '')),
                        
                        # Preferences
                        "location_preference": format_ai_data(ai_analysis.get('location_preference', '')),
                        "industry_preference": format_ai_data(ai_analysis.get('industry_preference', '')),
                        
                        # Salary
                        "salary_expectation": format_ai_data(ai_analysis.get('salary_expectation', '')),
                        "benefits_expectation": format_ai_data(ai_analysis.get('benefits_expectation', '')),
                        
                        # New fields
                        "primary_role": format_ai_data(ai_analysis.get('primary_role', '')),
                        "simple_search_terms": format_ai_data(ai_analysis.get('simple_search_terms', ''))
                    }
                    
                    analysis_complete = True
                    
                    # Store in session state
                    st.session_state.autofill_data = autofill_data
                    st.session_state.analysis_complete = True
                    st.session_state.ai_analysis = ai_analysis  # Save ai_analysis for later use

                    st.success("🎉 Resume analysis complete! Form has been auto-filled with your information.")

                except Exception as e:
                    st.error(f"❌ Error analyzing resume: {str(e)}")
                    st.stop()

    else:
        # Welcome screen
        st.info("📄 **Upload your CV above to get started!**")

        # Instructions
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            ### 📋 How it works:

            1. **📄 Upload** your CV (PDF or DOCX)
            2. **🤖 GPT-4** analyzes your skills, experience, and ideal roles
            3. **🔍 Search** LinkedIn jobs via RapidAPI (global search)
            4. **🎯 Rank** all jobs by match quality using AI
            5. **📊 See** your best matches with detailed scores!
            """)

        st.markdown("---")
        st.success("💡 **Pro Tip:** Jobs are searched globally (not filtered by Hong Kong) and ranked by how well they match your profile!")

    # ========== Form Area ==========
    if st.session_state.get('analysis_complete', False) or not cv_file:
        with st.form("job_seeker_form"):
            st.subheader("📝 Complete Your Profile")
            
            if st.session_state.get('analysis_complete', False):
                st.success("✅ Form auto-filled with your resume analysis!")
            
            st.markdown("Review and edit the auto-filled information from your CV analysis:")

            # Use data from session_state
            current_data = st.session_state.get('autofill_data', {})

            # Career Preferences - new fields at top of form
            st.subheader("🎯 Career Preferences")
            col_career1, col_career2 = st.columns(2)
            
            with col_career1:
                primary_role = st.text_input("Primary Role*", 
                                           value=current_data.get("primary_role", ""),
                                           placeholder="e.g., Project Manager, Software Engineer, Data Analyst")
            
            with col_career2:
                simple_search_terms = st.text_input("Search Keywords*", 
                                                  value=current_data.get("simple_search_terms", ""),
                                                  placeholder="e.g., python developer, project management, data science")

            # Educational background
            st.subheader("🎓 Educational background")
            col1, col2 = st.columns(2)

            with col1:
                education_options = ["Please select", "PhD", "Master", "Bachelor", "Diploma", "High School"]
                ed_level = current_data.get("education_level", "")
                education_index = smart_select_match(ed_level, education_options)
                
                education_level = st.selectbox(
                    "Educational level*",
                    education_options,
                    index=education_index
                )
                
                major = st.text_input("Major", 
                                    value=current_data.get("major", ""),
                                    placeholder="e.g., Computer Science, Business Administration")
                
                grad_options = ["Please select", "Graduated", "Fresh graduates", "Currently studying"]
                grad_status = current_data.get("graduation_status", "")
                grad_index = smart_select_match(grad_status, grad_options)
                
                graduation_status = st.selectbox(
                    "Graduation status*",
                    grad_options,
                    index=grad_index
                )

            with col2:
                uni_options = ["Please select", "985 Universities", "211 Universities", "Overseas Universities", "Regular Undergraduate Universities", "Other"]
                uni_bg = current_data.get("university_background", "")
                uni_index = smart_select_match(uni_bg, uni_options)
                
                university_background = st.selectbox(
                    "University background*",
                    uni_options,
                    index=uni_index
                )
                
                languages = st.text_input("Languages", 
                                        value=current_data.get("languages", ""),
                                        placeholder="e.g., English, Mandarin, Cantonese")
                
                certificates = st.text_input("Certificates", 
                                           value=current_data.get("certificates", ""),
                                           placeholder="e.g., PMP, CFA, AWS Certified")

            # Skills
            st.subheader("💼 Skills")
            hard_skills = st.text_area("Technical Skills", 
                                     value=current_data.get("hard_skills", ""),
                                     placeholder="e.g., Python, JavaScript, SQL, Machine Learning",
                                     height=100)
            
            soft_skills = st.text_area("Core Strengths", 
                                     value=current_data.get("soft_skills", ""),
                                     placeholder="e.g., Leadership, Communication, Problem Solving",
                                     height=100)

            # Work Experience
            st.subheader("📈 Work Experience")
            col3, col4 = st.columns(2)

            with col3:
                work_exp_options = ["Please select", "Recent Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"]
                work_exp = current_data.get("work_experience", "")
                work_index = smart_select_match(work_exp, work_exp_options)
                
                work_experience = st.selectbox(
                    "Work experience years*",
                    work_exp_options,
                    index=work_index
                )

            with col4:
                project_experience = st.text_area("Project experience", 
                                                value=current_data.get("project_experience", ""),
                                                placeholder="Describe your key projects and achievements",
                                                height=100)

            # Work preferences
            st.subheader("📍 Work preferences")
            col5, col6 = st.columns(2)

            with col5:
                loc_options = ["Please select", "Hong Kong", "Mainland China", "Overseas", "No Preference"]
                loc_pref = current_data.get("location_preference", "")
                loc_index = smart_select_match(loc_pref, loc_options)
                
                location_preference = st.selectbox(
                    "Location Preference*",
                    loc_options,
                    index=loc_index
                )
             
            with col6:
                industry_preference = st.text_input("Industry Preference", 
                                                  value=current_data.get("industry_preference", ""),
                                                  placeholder="e.g., Technology, Finance, Healthcare")
       
            # Salary and benefits expectations
            st.subheader("💰 Salary and Benefits Expectations")
            salary_expectation = st.text_input("Expected Salary Range", 
                                             value=current_data.get("salary_expectation", ""),
                                             placeholder="e.g., HKD 30,000 - 40,000")
            
            benefits_expectation = st.text_area("Benefits Requirements", 
                                              value=current_data.get("benefits_expectation", ""),
                                              placeholder="e.g., Medical insurance, Flexible working hours",
                                              height=80)
            

            # Submit button
            submitted = st.form_submit_button("💾 Save Information", use_container_width=True)

            if submitted:
                if (education_level == "Please select" or graduation_status == "Please select" or
                    university_background == "Please select" or work_experience == "Please select" or
                    location_preference == "Please select" or not primary_role.strip() or not simple_search_terms.strip()):
                    st.error("Please complete all required fields (marked with *)!")
                else:
                    # Save to database
                    job_seeker_id = save_job_seeker_info(
                        education_level, major, graduation_status, university_background,
                        languages, certificates, hard_skills, soft_skills, work_experience,
                        project_experience, location_preference, industry_preference,
                        salary_expectation, benefits_expectation,
                        primary_role,  # Use value from form
                        simple_search_terms  # Use value from form
                    )
                    
                    if job_seeker_id:
                        # Save to session state
                        st.session_state.job_seeker_id = job_seeker_id
                        st.success(f"✅ Information saved successfully! Your ID: {job_seeker_id}")
                        st.balloons()
                        
                        # Display success message
                        st.info(f"🔑 Your job seeker ID has been saved: **{job_seeker_id}**")
                        st.info("💡 You can use this ID on the Job Match page to view personalized job recommendations")
                    else:
                        st.error("❌ Failed to save information, please try again")

    """Save job seeker information to database"""

def job_recommendations_page(job_seeker_id=None):
    """Job Recommendations Page - Using Real API Data"""
    st.title("💼 Personalized Job Recommendations")

    # Get job seeker data - add error handling
    job_seeker_data = None
    try:
        if job_seeker_id:
            job_seeker_data = db.get_job_seeker_by_id(job_seeker_id)
        else:
            # If no ID provided, try to get latest record
            job_seeker_data = db.get_latest_job_seeker_data()
            
    except Exception as e:
        st.error(f"Error getting job seeker data: {e}")
        return

    if not job_seeker_data:
        st.error("No job seeker information found, please fill in your personal information first")
        st.info("Please fill in your information on the Job Seeker page")
        
        # Display debug information
        with st.expander("🔍 Debug Information"):
            st.write(f"Provided job_seeker_id: {job_seeker_id}")
            st.write("Trying to get latest record...")
            latest_id = db.get_latest_job_seeker_id()
            st.write(f"Latest record ID: {latest_id}")
            
        return

    # Display personal information summary
    with st.expander("👤 Your Personal Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Education:** {job_seeker_data.get('education_level', 'N/A')}")
            st.write(f"**Major:** {job_seeker_data.get('major', 'N/A')}")
            st.write(f"**Experience:** {job_seeker_data.get('work_experience', 'N/A')}")
            st.write(f"**Primary Role:** {job_seeker_data.get('primary_role', 'N/A')}")
        with col2:
            st.write(f"**Location Preference:** {job_seeker_data.get('location_preference', 'N/A')}")
            st.write(f"**Industry Preference:** {job_seeker_data.get('industry_preference', 'N/A')}")
            st.write(f"**Search Keywords:** {job_seeker_data.get('simple_search_terms', 'N/A')}")

    # Display skill information
    with st.expander("💼 Skill Information"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Technical Skills:**")
            hard_skills = job_seeker_data.get('hard_skills', '')
            if hard_skills:
                skills_list = [skill.strip() for skill in hard_skills.split(',')]
                for skill in skills_list[:10]:  # Show first 10 skills
                    st.write(f"• {skill}")
        with col2:
            st.write("**Core Strengths:**")
            soft_skills = job_seeker_data.get('soft_skills', '')
            if soft_skills:
                strengths_list = [strength.strip() for strength in soft_skills.split(',')]
                for strength in strengths_list[:5]:  # Show first 5 core strengths
                    st.write(f"• {strength}")

    # ----------------------------------------
    # 🔍 Job Search Settings
    # ----------------------------------------
    st.subheader("🔍 Job Search Settings")

    # Pre-fill defaults using job seeker data
    default_search = (
        job_seeker_data.get("primary_role", "")
        or job_seeker_data.get("simple_search_terms", "Python developer")
    )

    default_location = job_seeker_data.get("location_preference", "Hong Kong")

    col1, col2, col3 = st.columns(3)

    with col1:
        search_query = st.text_input(
            "Job Keywords*",
            value=default_search,
            placeholder="e.g.: software engineer, data analyst"
        )

    with col2:
        location = st.text_input(
            "City/Region",
            value=default_location,
            placeholder="e.g.: New York, London"
        )

    with col3:
        country = st.selectbox(
            "Country Code",
            ["hk", "us", "gb", "ca", "au", "sg"],
            index=0
        )

    col4, = st.columns(1)

    with col4:
        employment_types = st.multiselect(
            "Employment Type",
            ["FULLTIME", "PARTTIME", "CONTRACTOR"],
            default=["FULLTIME"]
        )


    # ----------------------------------------
    # 🔧 Advanced Search Tweaks
    # ----------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        num_jobs_to_search = st.slider(
            "Jobs to search", 
            10, 15, 5, 1,
            key="jobs_search_slider"
        )

    with col2:
        num_jobs_to_show = st.slider(
            "Top matches to display", 
            1, 10, 5,
            key="jobs_show_slider"
        )

    st.info(
        "💡 **Note:** Jobs are searched globally and ranked by how well they match your profile, regardless of location."
    )
    # -------------------------------------------------------
    # 🔎 STEP 2: Search Jobs via RapidAPI (SAFE VERSION)
    # -------------------------------------------------------
    with st.spinner(f"🔎 Step 2/3: Searching {num_jobs_to_search} jobs via RapidAPI..."):

        try:
            # ----------------------------------------------------
            # 1) Load job seeker ID safely
            # ----------------------------------------------------
            current_id = st.session_state.get("job_seeker_id")

            if not current_id:
                st.warning("⚠ job_seeker_id not found in session — using default search settings.")
                search_fields = {
                    "primary_role": "",
                    "simple_search_terms": "",
                    "location_preference": "Hong Kong",
                    "hard_skills": ""
                }
            else:
                # ----------------------------------------------------
                # 2) Load DB Search Fields
                # ----------------------------------------------------
                try:
                    search_fields = get_job_seeker_search_fields(current_id)
                except Exception as db_err:
                    st.error(f"❌ Database error when loading search settings: {db_err}")
                    search_fields = None

                if not search_fields:
                    st.warning("⚠ No stored search preferences found — using default search settings.")
                    search_fields = {
                        "primary_role": "",
                        "simple_search_terms": "",
                        "location_preference": "Hong Kong",
                        "hard_skills": ""
                    }

            # Extract fields
            primary_role        = search_fields.get("primary_role", "")
            simple_search_terms = search_fields.get("simple_search_terms", "")
            location_preference = search_fields.get("location_preference", "Hong Kong")
            hard_skills         = search_fields.get("hard_skills", "")

            # Construct resume_data with all fields
                        
            resume_data = {
                "education_level": job_seeker_data.get("education_level", ""),
                "major": job_seeker_data.get("major", ""),
                "graduation_status": job_seeker_data.get("graduation_status", ""),
                "university_background": job_seeker_data.get("university_background", ""),
                "languages": job_seeker_data.get("languages", ""),
                "certificates": job_seeker_data.get("certificates", ""),
                "hard_skills": job_seeker_data.get("hard_skills", ""),
                "soft_skills": job_seeker_data.get("soft_skills", ""),
                "work_experience": job_seeker_data.get("work_experience", ""),
                "project_experience": job_seeker_data.get("project_experience", ""),
                "location_preference": job_seeker_data.get("location_preference", ""),
                "industry_preference": job_seeker_data.get("industry_preference", ""),
                "salary_expectation": job_seeker_data.get("salary_expectation", ""),
                "benefits_expectation": job_seeker_data.get("benefits_expectation", ""),
                "primary_role": job_seeker_data.get("primary_role", ""),
                "simple_search_terms": job_seeker_data.get("simple_search_terms", ""),
            }

            # Construct ai_analysis dict, which can focus on skills, role, location, etc.
            ai_analysis = {
                "education_level": resume_data["education_level"],
                "major": resume_data["major"],
                "graduation_status": resume_data["graduation_status"],
                "university_background": resume_data["university_background"],
                "languages": [lang.strip() for lang in resume_data["languages"].split(",")] if resume_data["languages"] else [],
                "certificates": [cert.strip() for cert in resume_data["certificates"].split(",")] if resume_data["certificates"] else [],
                "skills": [skill.strip() for skill in resume_data["hard_skills"].split(",")] if resume_data["hard_skills"] else [],
                "soft_skills": [skill.strip() for skill in resume_data["soft_skills"].split(",")] if resume_data["soft_skills"] else [],
                "work_experience": resume_data["work_experience"],
                "project_experience": resume_data["project_experience"],
                "location_preference": resume_data["location_preference"],
                "industry_preference": resume_data["industry_preference"],
                "salary_expectation": resume_data["salary_expectation"],
                "benefits_expectation": resume_data["benefits_expectation"],
                "primary_role": resume_data["primary_role"],
                "simple_search_terms": resume_data["simple_search_terms"],
            }
    
            # ----------------------------------------------------
            # 3) Build search keyword string
            # ----------------------------------------------------
            search_keywords = ", ".join(
                field for field in [
                    primary_role,
                    simple_search_terms,
                    hard_skills, 
                ] if field.strip()
            )

            if not search_keywords:
                search_keywords = "General"

            # ----------------------------------------------------
            # 4) Show user what we are searching
            # ----------------------------------------------------
            st.info(
                f"📡 Searching LinkedIn via RapidAPI:\n\n"
                f"**Keywords:** {search_keywords}\n"
                f"**Location:** {location_preference}"
            )

            # ----------------------------------------------------
            # 5) Perform rapid API search
            # ----------------------------------------------------
            rapidapi = LinkedInJobSearcher(api_key=Config.RAPIDAPI_KEY)

            rapidapi_results = rapidapi.search_jobs(
                keywords=search_keywords,
                location=location_preference,
                limit=num_jobs_to_search
            )

            if not rapidapi_results:
                st.warning("⚠ No jobs found via RapidAPI. Try adjusting your keywords.")
                matched_jobs = []
            else:
                matched_jobs = rapidapi_results

        except Exception as e:
            st.error(f"❌ Unexpected error while searching jobs: {str(e)}")
            matched_jobs = []

        # ----------------------------------------
        # Step 2: Search and Match Jobs via Backend
        # ----------------------------------------
        with st.spinner(f"🔎 Step 2/3: Searching {num_jobs_to_search} jobs and matching..."):

            try:
                matched_jobs = backend.search_and_match_jobs(
                    resume_data=resume_data,
                    ai_analysis=ai_analysis,
                    num_jobs=num_jobs_to_search
                )
            except Exception as e:
                st.error(f"❌ Unexpected error while searching jobs: {str(e)}")
                st.stop()

        # ----------------------------------------
        # 📊 STEP 3: Display Results
        # ----------------------------------------
        st.markdown("---")

        if matched_jobs and len(matched_jobs) > 0:

            st.success(f"✅ Step 3/3: Found & ranked **{len(matched_jobs)}** jobs by match quality!")
            st.markdown(f"## 🎯 Top {num_jobs_to_show} Job Matches")

            st.info("📊 **Ranking Algorithm:** 60% Semantic Similarity + 40% Skill Match")

            # Display top matches
            for i, job in enumerate(matched_jobs[:num_jobs_to_show], start=1):

                combined = job.get("combined_score", 0)

                if combined >= 80:
                    match_emoji, match_label, match_color = "🟢", "Excellent Match", "#D4EDDA"
                elif combined >= 60:
                    match_emoji, match_label, match_color = "🟡", "Good Match", "#FFF3CD"
                else:
                    match_emoji, match_label, match_color = "🟠", "Fair Match", "#F8D7DA"

                expander_title = (
                    f"**#{i}** • {job.get('title', 'Unknown')} at {job.get('company', 'Unknown')} "
                    f"- {match_emoji} {match_label} ({combined:.1f}%)"
                )

                with st.expander(expander_title, expanded=i <= 2):

                    # Scores
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("🎯 Combined Score", f"{combined:.1f}%")
                    with col2:
                        st.metric("🧠 Semantic Match", f"{job.get('semantic_score', 0):.1f}%")
                    with col3:
                        st.metric("✅ Skill Match", f"{job.get('skill_match_percentage', 0):.1f}%")
                    with col4:
                        st.metric("🔢 Skills Matched", job.get("matched_skills_count", 0))

                    # Job details
                    st.markdown("##### 📋 Job Details")
                    detail_col1, detail_col2 = st.columns(2)

                    with detail_col1:
                        st.write(f"**📍 Location:** {job.get('location', 'Unknown')}")
                        st.write(f"**🏢 Company:** {job.get('company', 'Unknown')}")

                    with detail_col2:
                        st.write(f"**📅 Posted:** {job.get('posted_date', 'Unknown')}")
                        st.write(f"**💼 Role:** {job.get('title', 'Unknown')}")

                    # Matched skills (candidate has)
                    matched_skills = job.get("matched_skills", [])

                    # Required skills from job (assumes this field exists as a list)
                    required_skills = job.get("required_skills", [])

                    # Skills to improve: required but NOT matched
                    skills_to_improve = []
                    if required_skills:
                        required_set = set([s.lower() for s in required_skills])
                        matched_set = set([s.lower() for s in matched_skills])
                        missing_skills = required_set - matched_set
                        skills_to_improve = list(missing_skills)

                    # Display matched skills section
                    if matched_skills:
                        st.markdown("##### ✨ Your Skills That Match This Job")

                        badge_html = "".join(
                            f"""
                            <span style="
                                background-color:#D4EDDA;
                                color:#155724;
                                padding:5px 10px;
                                margin:3px;
                                border-radius:5px;
                                display:inline-block;
                                font-weight:bold;
                            ">✓ {skill}</span>
                            """
                            for skill in matched_skills[:8]
                        )

                        st.markdown(badge_html, unsafe_allow_html=True)

                        if len(matched_skills) > 8:
                            st.caption(f"+ {len(matched_skills) - 8} more matching skills")

                    # Display skills to improve section
                    if skills_to_improve:
                        st.markdown("##### 🛠 Skills You May Want to Improve")

                        badge_html_improve = "".join(
                            f"""
                            <span style="
                                background-color:#F8D7DA;
                                color:#721C24;
                                padding:5px 10px;
                                margin:3px;
                                border-radius:5px;
                                display:inline-block;
                                font-weight:bold;
                            ">✗ {skill}</span>
                            """
                            for skill in skills_to_improve[:8]
                        )

                        st.markdown(badge_html_improve, unsafe_allow_html=True)

                        if len(skills_to_improve) > 8:
                            st.caption(f"+ {len(skills_to_improve) - 8} more skills to consider")

                    # Description
                    description = job.get("description", "")
                    if description:
                        st.markdown("##### 📝 Job Description")
                        preview = description[:500]
                        st.text_area(
                            "Preview",
                            preview + ("..." if len(description) > 500 else ""),
                            height=120,
                            key=f"desc_{job.get('id', i)}"
                        )

                    # Apply link
                    job_url = job.get("url", "")
                    if job_url:
                        st.link_button(
                            "🔗 Apply Now on LinkedIn",
                            job_url,
                            use_container_width=True,
                            type="primary"
                        )
                    else:
                        st.info("🔗 Application link not available")

        else:
            st.warning("⚠️ No matched jobs found. Please try adjusting your search criteria.")

def enhanced_head_hunter_page():
    """Enhanced Head Hunter Page - Job Publishing and Management"""
    st.title("🎯 Head Hunter Portal")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Publish New Position", "View Published Positions", "Position Statistics"]
    )

    if page_option == "Publish New Position":
        publish_new_job()
    elif page_option == "View Published Positions":
        view_published_jobs()
    elif page_option == "Position Statistics":
        show_job_statistics()

def publish_new_job():
    """Publish New Position Form"""
    st.header("📝 Publish New Position")

    with st.form("head_hunter_job_form"):
        # Basic Position Information
        st.subheader("🎯 Basic Position Information")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("Position Title*", placeholder="e.g.: Senior Frontend Engineer")
        with col2:
            employment_type = st.selectbox("Employment Type*", ["Please select", "Full-time", "Part-time", "Contract", "Internship"])

        job_description = st.text_area("Job Description*", height=100,
                                      placeholder="Detailed introduction of position main content and team situation...")

        main_responsibilities = st.text_area("Main Responsibilities*", height=100,
                                           placeholder="List main responsibilities with bullet points, one per line...")

        required_skills = st.text_area("Required Skills & Qualifications*", height=100,
                                     placeholder="e.g.: 5+ years experience, proficient in React.js, Computer Science degree...")

        # Company and Client Information
        st.subheader("🏢 Company and Client Information")

        col3, col4 = st.columns(2)
        with col3:
            client_company = st.text_input("Client Company Name*", placeholder="Company official name")
            industry = st.selectbox("Industry*", ["Please select", "Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])
        with col4:
            work_location = st.selectbox("Work Location*", ["Please select", "Hong Kong", "Mainland China", "Overseas", "Remote"])
            company_size = st.selectbox("Company Size*", ["Please select", "Startup (1-50)", "SME (51-200)", "Large Enterprise (201-1000)", "Multinational (1000+)"])

        work_type = st.selectbox("Work Type*", ["Please select", "Remote", "Hybrid", "Office"])

        # Employment Details
        st.subheader("💼 Employment Details")

        col5, col6 = st.columns(2)
        with col5:
            experience_level = st.selectbox("Experience Level*", ["Please select", "Fresh Graduate", "1-3 years", "3-5 years", "5-10 years", "10+ years"])
        with col6:
            visa_support = st.selectbox("Visa Support", ["Not provided", "Work Visa", "Assistance provided", "Must have own visa"])

        # Salary and Application Method
        st.subheader("💰 Salary and Application Method")

        col7, col8, col9 = st.columns([2, 2, 1])
        with col7:
            min_salary = st.number_input("Minimum Salary*", min_value=0, value=30000, step=5000)
        with col8:
            max_salary = st.number_input("Maximum Salary*", min_value=0, value=50000, step=5000)
        with col9:
            currency = st.selectbox("Currency", ["HKD", "USD", "CNY", "EUR", "GBP"])

        benefits = st.text_area("Benefits", height=80,
                              placeholder="e.g.: Medical insurance, 15 days annual leave, performance bonus, stock options...")

        application_method = st.text_area("Application Method*", height=80,
                                        value="Please send resume to recruit@headhunter.com, include position title in email subject",
                                        placeholder="Application process and contact information...")

        job_valid_until = st.date_input("Position Posting Validity Period*",
                                      value=datetime.now().date() + pd.Timedelta(days=30))

        # Submit button
        submitted = st.form_submit_button("💾 Publish Position", type="primary", use_container_width=True)

        if submitted:
            # Validate required fields
            required_fields = [
                job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, min_salary, max_salary, application_method
            ]

            if "Please select" in [employment_type, industry, work_location, work_type, company_size, experience_level]:
                st.error("Please complete all required fields (marked with *)!")
            elif not all(required_fields):
                st.error("Please complete all required fields (marked with *)!")
            elif min_salary >= max_salary:
                st.error("Maximum salary must be greater than minimum salary!")
            
            # Modify this part in Streamlit app:
            else:
                # Create dictionary object
                job_data = {
                    'job_title': job_title,
                    'job_description': job_description,
                    'main_responsibilities': main_responsibilities,
                    'required_skills': required_skills,
                    'client_company': client_company,
                    'industry': industry,
                    'work_location': work_location,
                    'work_type': work_type,
                    'company_size': company_size,
                    'employment_type': employment_type,
                    'experience_level': experience_level,
                    'visa_support': visa_support,
                    'min_salary': min_salary,
                    'max_salary': max_salary,
                    'currency': currency,
                    'benefits': benefits,
                    'application_method': application_method,
                    'job_valid_until': job_valid_until.strftime("%Y-%m-%d")
                }
                
                # Save to database - now only pass one parameter
                success = save_head_hunter_job(job_data)

                if success:
                    st.success("✅ Position published successfully!")
                    st.balloons()
                else:
                    st.error("❌ Position publishing failed, please try again")


def view_published_jobs():
    """View Published Positions"""
    st.header("📋 Published Positions")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No positions published yet")
        return

    st.success(f"Published {len(jobs)} positions")

    # Search and filter
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("Search position title or company")
    with col2:
        filter_industry = st.selectbox("Filter by industry", ["All industries"] + ["Technology", "Finance", "Consulting", "Healthcare", "Education", "Manufacturing", "Retail", "Other"])

    # Filter positions
    filtered_jobs = jobs
    if search_term:
        filtered_jobs = [job for job in jobs if search_term.lower() in job[2].lower() or search_term.lower() in job[6].lower()]
    if filter_industry != "All industries":
        filtered_jobs = [job for job in filtered_jobs if job[7] == filter_industry]

    if not filtered_jobs:
        st.warning("No matching positions found")
        return

    # Display position list
    for job in filtered_jobs:
        with st.expander(f"#{job[0]} {job[2]} - {job[6]}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Published Time:** {job[1]}")
                st.write(f"**Company:** {job[6]}")
                st.write(f"**Industry:** {job[7]}")
                st.write(f"**Location:** {job[8]} ({job[9]})")
                st.write(f"**Size:** {job[10]}")

            with col2:
                st.write(f"**Type:** {job[11]}")
                st.write(f"**Experience:** {job[12]}")
                st.write(f"**Salary:** {job[14]:,} - {job[15]:,} {job[16]}")
                st.write(f"**Valid Until:** {job[19]}")
                if job[13] != "Not provided":
                    st.write(f"**Visa:** {job[13]}")

            st.write("**Description:**")
            st.write(job[3][:200] + "..." if len(job[3]) > 200 else job[3])

def show_job_statistics():
    """Display Position Statistics"""
    st.header("📊 Position Statistics")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("No statistics available yet")
        return

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Positions", len(jobs))
    with col2:
        active_jobs = len([job for job in jobs if datetime.strptime(job[19], "%Y-%m-%d").date() >= datetime.now().date()])
        st.metric("Active Positions", active_jobs)
    with col3:
        expired_jobs = len(jobs) - active_jobs
        st.metric("Expired Positions", expired_jobs)
    with col4:
        avg_salary = sum((job[14] + job[15]) / 2 for job in jobs) / len(jobs)
        st.metric("Average Salary", f"{avg_salary:,.0f}")

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[7]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        st.write(f"• **{industry}:** {count} positions ({count/len(jobs)*100:.1f}%)")

    # Location distribution
    st.subheader("📍 Work Location Distribution")
    location_counts = {}
    for job in jobs:
        location = job[8]
        location_counts[location] = location_counts.get(location, 0) + 1

    for location, count in location_counts.items():
        st.write(f"• **{location}:** {count} positions")

    # Experience requirement distribution
    st.subheader("🎯 Experience Requirement Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[12]
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for experience, count in experience_counts.items():
        st.write(f"• **{experience}:** {count} positions")

def recruitment_match_dashboard():
    """Recruitment Match Dashboard"""
    st.title("🎯 Recruitment Match Portal")

    # Quick statistics
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Positions", len(jobs) if jobs else 0)
    with col2:
        st.metric("Job Seekers", len(seekers) if seekers else 0)
    with col3:
        st.metric("Match Ready", "✅" if jobs and seekers else "❌")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Smart Talent Matching", "Match Statistics", "Instructions"]
    )

    if page_option == "Smart Talent Matching":
        recruitment_match_page()
    elif page_option == "Match Statistics":
        show_match_statistics()
    else:
        show_instructions()

def recruitment_match_page():
    """Recruitment Match Page"""
    st.title("🎯 Recruitment Match - Smart Talent Matching")

    # Get data
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    if not jobs:
        st.warning("❌ No available position information, please first publish positions in the headhunter module")
        return

    if not seekers:
        st.warning("❌ No available job seeker information, please first fill in information on Job Seeker page")
        return

    st.success(f"📊 System has {len(jobs)} active positions and {len(seekers)} job seekers")

    # Select position for matching
    st.subheader("🔍 Select Position to Match")

    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("Select Position", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # Display position details
    with st.expander("📋 Position Details", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Position ID:** #{selected_job[0]}")
            st.write(f"**Company:** {selected_job[5]}")
            st.write(f"**Industry:** {selected_job[6]}")
            st.write(f"**Experience Requirement:** {selected_job[11]}")
        with col2:
            st.write(f"**Location:** {selected_job[7]}")
            st.write(f"**Salary:** {selected_job[13]:,}-{selected_job[14]:,} {selected_job[15]}")
            st.write(f"**Skill Requirements:** {selected_job[4][:100]}...")

    # Match options
    st.subheader("⚙️ Match Settings")
    col1, col2 = st.columns(2)
    with col1:
        min_match_score = st.slider("Minimum Match Score", 0, 100, 60)
    with col2:
        max_candidates = st.slider("Display Top N Candidates", 1, 20, 10)

    # Execute matching
    if st.button("🚀 Start Smart Matching", type="primary", use_container_width=True):
        st.subheader("📈 Match Results")

        progress_bar = st.progress(0)
        results = []

        for i, seeker in enumerate(seekers[:max_candidates]):
            progress = (i + 1) / min(len(seekers), max_candidates)
            progress_bar.progress(progress)

            # Use simplified matching algorithm
            analysis_result = analyze_match_simple(selected_job, seeker)
            match_score = analysis_result.get('match_score', 0)

            if match_score >= min_match_score:
                results.append({
                    'seeker_id': seeker[0],
                    'name': seeker[1],
                    'current_title': seeker[9],
                    'experience': seeker[3],
                    'education': seeker[4],
                    'match_score': match_score,
                    'analysis': analysis_result,
                    'raw_data': seeker
                })

        progress_bar.empty()

        # Display results
        if results:
            results.sort(key=lambda x: x['match_score'], reverse=True)
            st.success(f"🎉 Found {len(results)} matching candidates (score ≥ {min_match_score})")

            for i, result in enumerate(results):
                score_color = "🟢" if result['match_score'] >= 80 else "🟡" if result['match_score'] >= 60 else "🔴"

                with st.expander(f"{score_color} #{i+1} {result['name']} - {result['match_score']} points", expanded=i < 2):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**Candidate Information:**")
                        st.write(f"**ID:** #{result['seeker_id']}")
                        st.write(f"**Education Background:** {result['education']}")
                        st.write(f"**Work Experience:** {result['experience']}")
                        st.write(f"**Current Background:** {result['current_title']}")
                        st.write(f"**Skills:** {result['raw_data'][2][:100]}...")

                    with col2:
                        st.write("**Match Analysis:**")
                        st.write(f"**Match Score:** {score_color} {result['match_score']} points")
                        st.write(f"**Salary Match:** {result['analysis'].get('salary_match', 'Average')}")
                        st.write(f"**Culture Fit:** {result['analysis'].get('culture_fit', 'Medium')}")

                        if 'key_strengths' in result['analysis']:
                            st.write("**Core Strengths:**")
                            for strength in result['analysis']['key_strengths']:
                                st.write(f"✅ {strength}")

                        if 'potential_gaps' in result['analysis']:
                            st.write("**Areas of Concern:**")
                            for gap in result['analysis']['potential_gaps']:
                                st.write(f"⚠️ {gap}")

                    if 'recommendation' in result['analysis']:
                        st.info(f"**Recommendation:** {result['analysis']['recommendation']}")

                    # Action buttons
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📞 Contact Candidate", key=f"contact_{result['seeker_id']}"):
                            st.success(f"Marked for contact: {result['name']}")
                    with col_btn2:
                        if st.button("💼 Schedule Interview", key=f"interview_{result['seeker_id']}"):
                            st.success(f"Interview scheduled: {result['name']}")
        else:
            st.warning("😔 No matching candidates found, please adjust matching conditions")

def ai_interview_dashboard():
    """AI Interview Dashboard"""
    st.title("🤖 AI Mock Interview System")

    # Quick statistics
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Available Positions", len(jobs) if jobs else 0)
    with col2:
        st.metric("Personal Profile", "✅" if seeker_profile else "❌")
    with col3:
        if 'interview' in st.session_state:
            progress = st.session_state.interview['current_question']
            total = st.session_state.interview['total_questions']
            st.metric("Interview Progress", f"{progress}/{total}")
        else:
            st.metric("Interview Status", "Not Started")

    # Page selection
    page_option = st.sidebar.radio(
        "Select Function",
        ["Start Mock Interview", "Interview Preparation Guide", "Instructions"]
    )

    if page_option == "Start Mock Interview":
        ai_interview_page()
    elif page_option == "Interview Preparation Guide":
        show_interview_guidance()
    else:
        show_interview_instructions()

def show_interview_guidance():
    """Display Interview Preparation Guide"""
    st.header("🎯 Interview Preparation Guide")

    st.info("""
    **Interview Preparation Suggestions:**

    ### 📚 Technical Interview Preparation
    1. **Review Core Skills**: Ensure mastery of key technologies required for the position
    2. **Prepare Project Cases**: Prepare 2-3 projects that demonstrate your capabilities
    3. **Practice Coding Problems**: Prepare algorithms and data structures for technical positions

    ### 💼 Behavioral Interview Preparation
    1. **STAR Method**: Situation-Task-Action-Result
    2. **Prepare Success Stories**: Show how you solve problems and create value
    3. **Understand Company Culture**: Research company values and work style

    ### 🎯 Communication Skills
    1. **Clear Expression**: Structure your answers
    2. **Active Listening**: Ensure understanding of question core
    3. **Show Enthusiasm**: Express interest in position and company
    """)

def show_interview_instructions():
    """Display Usage Instructions"""
    st.header("📖 AI Mock Interview Usage Instructions")

    st.info("""
    **AI Mock Interview Function Guide:**

    ### 🚀 Start Interview
    1. **Select Position**: Choose a position from headhunter published positions for mock interview
    2. **Start Interview**: AI will generate relevant questions based on position requirements
    3. **Answer Questions**: Provide detailed answers for each question

    ### 📊 Interview Process
    - **10 Questions**: Includes various types like technical, behavioral, situational
    - **Real-time Evaluation**: AI evaluates quality of each answer
    - **Personalized Questions**: Follow-up questions based on your previous answers

    ### 🎯 Get Feedback
    - **Detailed Scoring**: Specific scoring and feedback for each question
    - **Overall Evaluation**: Complete interview performance summary
    - **Improvement Suggestions**: Targeted career development advice

    **Tip**: Please ensure use in stable network environment for AI to generate questions and evaluate answers normally.
    """)

# Add debug tools in sidebar
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 Database Debug")
    
    if st.button("View All Job Seeker Records"):
        try:
            conn = sqlite3.connect('job_seeker.db')
            c = conn.cursor()
            c.execute("SELECT job_seeker_id, timestamp, education_level, primary_role FROM job_seekers ORDER BY id DESC")
            results = c.fetchall()
            conn.close()
            
            if results:
                st.write("📋 All Job Seeker Records:")
                for record in results:
                    st.write(f"- ID: {record[0]}, Time: {record[1]}, Education: {record[2]}, Role: {record[3]}")
            else:
                st.write("No job seeker records yet")
        except Exception as e:
            st.error(f"Query failed: {e}")
    
    # Display current session state
    current_id = st.session_state.get('job_seeker_id')
    if current_id:
        st.info(f"Current Session ID: **{current_id}**")

# Sidebar navigation
st.sidebar.title("🔍 Navigation")

# Navigation buttons
if st.sidebar.button("🏠 Job Seeker", use_container_width=True, key="main_btn"):
    st.session_state.current_page = "main"
if st.sidebar.button("💼 Job Match", use_container_width=True):
    st.session_state.current_page = "job_recommendations"
if st.sidebar.button("🎯 Recruiter", use_container_width=True):
        st.session_state.current_page = "head_hunter"
if st.sidebar.button("🔍 Recruitment Match", use_container_width=True):
        st.session_state.current_page = "recruitment_match"
if st.sidebar.button("🤖 AI Interview", use_container_width=True):
        st.session_state.current_page = "ai_interview"

# Page routing
if st.session_state.current_page == "main":
    main_analyzer_page()
elif st.session_state.current_page == "job_recommendations":
    job_seeker_id = st.session_state.get('job_seeker_id')

    # Check if there is saved job seeker data
    if not job_seeker_id:
        st.warning("⚠️ Please first save your personal information on the Job Seeker page")
        st.info("👉 Switch to 'Job Seeker' page to fill in and save your information")
        
        # Provide quick jump
        if st.button("Go to Job Seeker Page"):
            st.session_state.current_page = "main"
            st.rerun()
    else:
        # Call job recommendations page function
        job_recommendations_page(job_seeker_id)

elif st.session_state.current_page == "head_hunter":
    enhanced_head_hunter_page()
elif st.session_state.current_page == "recruitment_match":
    recruitment_match_dashboard()
elif st.session_state.current_page == "ai_interview":
    ai_interview_dashboard()


# Sidebar information
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 Usage Instructions

1. **Home**: Smart Resume-JD Matching Analyzer
2. **Job Seeker**: Fill information → Automatic job recommendations
3. **Job Match**: View AI-matched positions
4. **Head Hunter**: Publish and manage recruitment positions
5. **Recruitment Match**: Smart candidate-position matching
6. **AI Interview**: Mock interviews and skill assessment
7. **DB Verify**: Verify data storage
""")
                    
# Footer
st.markdown("---")
st.caption("🤖 Powered by GPT-4, Pinecone Vector Search, and RapidAPI LinkedIn Jobs")

# Application startup
if __name__ == "__main__":
    # Ensure application runs normally
    pass