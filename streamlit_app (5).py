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
    """主页 - Smart Career"""
    st.title("🎯 Smart Career")
    st.markdown("Upload your CV and let **GPT-4** find matching jobs globally, ranked by match quality!")

    # 定义辅助函数
    def smart_select_match(value, options):
        """智能匹配选择框选项"""
        if not value:
            return 0
        
        value_str = str(value).lower()
        for i, option in enumerate(options):
            if option.lower() in value_str or value_str in option.lower():
                return i
        return 0

    def format_ai_data(data, default=""):
        """格式化AI返回的数据"""
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
    ai_analysis = {}  # 初始化 ai_analysis

    if cv_file:
        st.success(f"✅ Uploaded: **{cv_file.name}**")

        if st.button("🔍 Analyze with GPT-4", type="primary", use_container_width=True, key="analyze_button"):

            # STEP 1: Analyze Resume
            with st.spinner("🤖 Step 1/2: Analyzing your resume with GPT-4..."):
                try:
                    resume_data, ai_analysis = backend.process_resume(cv_file, cv_file.name)
                    
                    st.balloons()

                    # 展示分析結果
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

                    # 提取并格式化数据
                    autofill_data = {
                        # 教育背景
                        "education_level": format_ai_data(ai_analysis.get('education_level', '')),
                        "major": format_ai_data(ai_analysis.get('major', '')),
                        "graduation_status": format_ai_data(ai_analysis.get('graduation_status', '')),
                        "university_background": format_ai_data(ai_analysis.get('university_background', '')),
                        
                        # 语言和证书
                        "languages": format_ai_data(ai_analysis.get('languages', '')),
                        "certificates": format_ai_data(ai_analysis.get('certificates', '')),
                        
                        # 技能 - 直接使用检测到的技能
                        "hard_skills": format_ai_data(skills),  # 使用检测到的技能
                        "soft_skills": format_ai_data(ai_analysis.get('core_strengths', [])),  # 使用核心优势
                        
                        # 工作经验
                        "work_experience": format_ai_data(ai_analysis.get('work_experience', '')),
                        "project_experience": format_ai_data(ai_analysis.get('project_experience', '')),
                        
                        # 偏好
                        "location_preference": format_ai_data(ai_analysis.get('location_preference', '')),
                        "industry_preference": format_ai_data(ai_analysis.get('industry_preference', '')),
                        
                        # 薪资
                        "salary_expectation": format_ai_data(ai_analysis.get('salary_expectation', '')),
                        "benefits_expectation": format_ai_data(ai_analysis.get('benefits_expectation', '')),
                        
                        # 新增字段
                        "primary_role": format_ai_data(ai_analysis.get('primary_role', '')),
                        "simple_search_terms": format_ai_data(ai_analysis.get('simple_search_terms', ''))
                    }
                    
                    analysis_complete = True
                    
                    # 存储到session state
                    st.session_state.autofill_data = autofill_data
                    st.session_state.analysis_complete = True
                    st.session_state.ai_analysis = ai_analysis  # 保存ai_analysis供后续使用

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

    # ========== 表单区域 ==========
    if st.session_state.get('analysis_complete', False) or not cv_file:
        with st.form("job_seeker_form"):
            st.subheader("📝 Complete Your Profile")
            
            if st.session_state.get('analysis_complete', False):
                st.success("✅ Form auto-filled with your resume analysis!")
            
            st.markdown("Review and edit the auto-filled information from your CV analysis:")

            # 使用session_state中的数据
            current_data = st.session_state.get('autofill_data', {})

            # 职业偏好 - 新增字段放在表单顶部
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

            # 教育背景
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

            # 技能
            st.subheader("💼 Skills")
            hard_skills = st.text_area("Technical Skills", 
                                     value=current_data.get("hard_skills", ""),
                                     placeholder="e.g., Python, JavaScript, SQL, Machine Learning",
                                     height=100)
            
            soft_skills = st.text_area("Core Strengths", 
                                     value=current_data.get("soft_skills", ""),
                                     placeholder="e.g., Leadership, Communication, Problem Solving",
                                     height=100)

            # 工作经验
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

            # 工作偏好
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
       
            # 薪资福利期望
            st.subheader("💰 Salary and Benefits Expectations")
            salary_expectation = st.text_input("Expected Salary Range", 
                                             value=current_data.get("salary_expectation", ""),
                                             placeholder="e.g., HKD 30,000 - 40,000")
            
            benefits_expectation = st.text_area("Benefits Requirements", 
                                              value=current_data.get("benefits_expectation", ""),
                                              placeholder="e.g., Medical insurance, Flexible working hours",
                                              height=80)
            

            # 提交按钮
            submitted = st.form_submit_button("💾 Save Information", use_container_width=True)

            if submitted:
                if (education_level == "Please select" or graduation_status == "Please select" or
                    university_background == "Please select" or work_experience == "Please select" or
                    location_preference == "Please select" or not primary_role.strip() or not simple_search_terms.strip()):
                    st.error("Please complete all required fields (marked with *)!")
                else:
                    # 保存到数据库
                    job_seeker_id = save_job_seeker_info(
                        education_level, major, graduation_status, university_background,
                        languages, certificates, hard_skills, soft_skills, work_experience,
                        project_experience, location_preference, industry_preference,
                        salary_expectation, benefits_expectation,
                        primary_role,  # 使用表单中的值
                        simple_search_terms  # 使用表单中的值
                    )
                    
                    if job_seeker_id:
                        # 保存到session state
                        st.session_state.job_seeker_id = job_seeker_id
                        st.success(f"✅ Information saved successfully! Your ID: {job_seeker_id}")
                        st.balloons()
                        
                        # 显示成功信息
                        st.info(f"🔑 您的求职者ID已保存: **{job_seeker_id}**")
                        st.info("💡 您可以在 Job Match 页面使用此ID查看个性化职位推荐")
                    else:
                        st.error("❌ Failed to save information, please try again")

    """保存求职者信息到数据库"""

def job_recommendations_page(job_seeker_id=None):
    """职位推荐页面 - 使用真实API数据"""
    st.title("💼 个性化职位推荐")

    # 获取求职者数据 - 添加错误处理
    job_seeker_data = None
    try:
        if job_seeker_id:
            job_seeker_data = db.get_job_seeker_by_id(job_seeker_id)
        else:
            # 如果没有提供ID，尝试获取最新记录
            job_seeker_data = db.get_latest_job_seeker_data()
            
    except Exception as e:
        st.error(f"获取求职者数据时出错: {e}")
        return

    if not job_seeker_data:
        st.error("未找到求职者信息，请先填写个人信息")
        st.info("请在 Job Seeker 页面填写您的信息")
        
        # 显示调试信息
        with st.expander("🔍 调试信息"):
            st.write(f"提供的 job_seeker_id: {job_seeker_id}")
            st.write("尝试获取最新记录...")
            latest_id = db.get_latest_job_seeker_id()
            st.write(f"最新记录ID: {latest_id}")
            
        return

    # 显示个人信息摘要
    with st.expander("👤 您的个人信息"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**学历:** {job_seeker_data.get('education_level', 'N/A')}")
            st.write(f"**专业:** {job_seeker_data.get('major', 'N/A')}")
            st.write(f"**经验:** {job_seeker_data.get('work_experience', 'N/A')}")
            st.write(f"**主要角色:** {job_seeker_data.get('primary_role', 'N/A')}")
        with col2:
            st.write(f"**地点偏好:** {job_seeker_data.get('location_preference', 'N/A')}")
            st.write(f"**行业偏好:** {job_seeker_data.get('industry_preference', 'N/A')}")
            st.write(f"**搜索关键词:** {job_seeker_data.get('simple_search_terms', 'N/A')}")

    # 显示技能信息
    with st.expander("💼 技能信息"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**技术技能:**")
            hard_skills = job_seeker_data.get('hard_skills', '')
            if hard_skills:
                skills_list = [skill.strip() for skill in hard_skills.split(',')]
                for skill in skills_list[:10]:  # 显示前10个技能
                    st.write(f"• {skill}")
        with col2:
            st.write("**核心优势:**")
            soft_skills = job_seeker_data.get('soft_skills', '')
            if soft_skills:
                strengths_list = [strength.strip() for strength in soft_skills.split(',')]
                for strength in strengths_list[:5]:  # 显示前5个核心优势
                    st.write(f"• {strength}")

    # ----------------------------------------
    # 🔍 Job Search Settings
    # ----------------------------------------
    st.subheader("🔍 搜索职位设置")

    # Pre-fill defaults using job seeker data
    default_search = (
        job_seeker_data.get("primary_role", "")
        or job_seeker_data.get("simple_search_terms", "Python developer")
    )

    default_location = job_seeker_data.get("location_preference", "Hong Kong")

    col1, col2, col3 = st.columns(3)

    with col1:
        search_query = st.text_input(
            "职位关键词*",
            value=default_search,
            placeholder="例如: software engineer, data analyst"
        )

    with col2:
        location = st.text_input(
            "城市/地区",
            value=default_location,
            placeholder="例如: New York, London"
        )

    with col3:
        country = st.selectbox(
            "国家代码",
            ["hk", "us", "gb", "ca", "au", "sg"],
            index=0
        )

    col4, = st.columns(1)

    with col4:
        employment_types = st.multiselect(
            "工作类型",
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
    """增强的猎头页面 - 职位发布和管理"""
    st.title("🎯 Head Hunter Portal")

    # 页面选择
    page_option = st.sidebar.radio(
        "选择功能",
        ["发布新职位", "查看已发布职位", "职位统计"]
    )

    if page_option == "发布新职位":
        publish_new_job()
    elif page_option == "查看已发布职位":
        view_published_jobs()
    elif page_option == "职位统计":
        show_job_statistics()

def publish_new_job():
    """发布新职位表单"""
    st.header("📝 发布新职位")

    with st.form("head_hunter_job_form"):
        # 职位基本信息
        st.subheader("🎯 职位基本信息")

        col1, col2 = st.columns(2)
        with col1:
            job_title = st.text_input("职位标题*", placeholder="例如：高级前端工程师")
        with col2:
            employment_type = st.selectbox("雇佣类型*", ["请选择", "全职", "兼职", "合同", "实习"])

        job_description = st.text_area("职位描述*", height=100,
                                      placeholder="详细介绍职位的主要内容和团队情况...")

        main_responsibilities = st.text_area("主要职责*", height=100,
                                           placeholder="用要点列出主要职责，每行一个职责...")

        required_skills = st.text_area("必备技能与资格*", height=100,
                                     placeholder="例如：5年以上经验，精通React.js，计算机科学学位...")

        # 公司与客户信息
        st.subheader("🏢 公司与客户信息")

        col3, col4 = st.columns(2)
        with col3:
            client_company = st.text_input("客户公司名称*", placeholder="公司官方名称")
            industry = st.selectbox("行业*", ["请选择", "科技", "金融", "咨询", "医疗", "教育", "制造", "零售", "其他"])
        with col4:
            work_location = st.selectbox("工作地点*", ["请选择", "香港", "内地", "海外", "远程"])
            company_size = st.selectbox("公司规模*", ["请选择", "初创公司(1-50)", "中小型企业(51-200)", "大型企业(201-1000)", "跨国公司(1000+)"])

        work_type = st.selectbox("工作类型*", ["请选择", "远程", "混合", "办公室"])

        # 雇佣详情
        st.subheader("💼 雇佣详情")

        col5, col6 = st.columns(2)
        with col5:
            experience_level = st.selectbox("经验级别*", ["请选择", "应届", "1-3年", "3-5年", "5-10年", "10年以上"])
        with col6:
            visa_support = st.selectbox("签证支持", ["不提供", "工作签证", "协助办理", "需自有签证"])

        # 薪酬与申请方式
        st.subheader("💰 薪酬与申请方式")

        col7, col8, col9 = st.columns([2, 2, 1])
        with col7:
            min_salary = st.number_input("最低薪资*", min_value=0, value=30000, step=5000)
        with col8:
            max_salary = st.number_input("最高薪资*", min_value=0, value=50000, step=5000)
        with col9:
            currency = st.selectbox("货币", ["HKD", "USD", "CNY", "EUR", "GBP"])

        benefits = st.text_area("福利待遇", height=80,
                              placeholder="例如：医疗保险、年假15天、绩效奖金、股票期权...")

        application_method = st.text_area("申请方式*", height=80,
                                        value="请将简历发送至 recruit@headhunter.com，邮件标题请注明申请职位",
                                        placeholder="申请流程和联系方式...")

        job_valid_until = st.date_input("职位发布有效期*",
                                      value=datetime.now().date() + pd.Timedelta(days=30))

        # 提交按钮
        submitted = st.form_submit_button("💾 发布职位", type="primary", use_container_width=True)

        if submitted:
            # 验证必填字段
            required_fields = [
                job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, min_salary, max_salary, application_method
            ]

            if "请选择" in [employment_type, industry, work_location, work_type, company_size, experience_level]:
                st.error("请完成所有必填字段（标*号）！")
            elif not all(required_fields):
                st.error("请完成所有必填字段（标*号）！")
            elif min_salary >= max_salary:
                st.error("最高薪资必须大于最低薪资！")
            
            # 在 Streamlit app 中修改这部分代码：
            else:
                # 创建字典对象
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
                
                # 保存到数据库 - 现在只传递一个参数
                success = save_head_hunter_job(job_data)

                if success:
                    st.success("✅ 职位发布成功！")
                    st.balloons()
                else:
                    st.error("❌ 职位发布失败，请重试")


def view_published_jobs():
    """查看已发布的职位"""
    st.header("📋 已发布职位")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("尚未发布任何职位")
        return

    st.success(f"已发布 {len(jobs)} 个职位")

    # 搜索和筛选
    col1, col2 = st.columns(2)
    with col1:
        search_term = st.text_input("搜索职位标题或公司")
    with col2:
        filter_industry = st.selectbox("按行业筛选", ["所有行业"] + ["科技", "金融", "咨询", "医疗", "教育", "制造", "零售", "其他"])

    # 过滤职位
    filtered_jobs = jobs
    if search_term:
        filtered_jobs = [job for job in jobs if search_term.lower() in job[2].lower() or search_term.lower() in job[6].lower()]
    if filter_industry != "所有行业":
        filtered_jobs = [job for job in filtered_jobs if job[7] == filter_industry]

    if not filtered_jobs:
        st.warning("没有找到匹配的职位")
        return

    # 显示职位列表
    for job in filtered_jobs:
        with st.expander(f"#{job[0]} {job[2]} - {job[6]}", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**发布时间:** {job[1]}")
                st.write(f"**公司:** {job[6]}")
                st.write(f"**行业:** {job[7]}")
                st.write(f"**地点:** {job[8]} ({job[9]})")
                st.write(f"**规模:** {job[10]}")

            with col2:
                st.write(f"**类型:** {job[11]}")
                st.write(f"**经验:** {job[12]}")
                st.write(f"**薪资:** {job[14]:,} - {job[15]:,} {job[16]}")
                st.write(f"**有效期:** {job[19]}")
                if job[13] != "不提供":
                    st.write(f"**签证:** {job[13]}")

            st.write("**描述:**")
            st.write(job[3][:200] + "..." if len(job[3]) > 200 else job[3])

def show_job_statistics():
    """显示职位统计"""
    st.header("📊 职位统计")

    jobs = db2.get_all_head_hunter_jobs()

    if not jobs:
        st.info("尚无统计数据")
        return

    # 基本统计
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("总职位数", len(jobs))
    with col2:
        active_jobs = len([job for job in jobs if datetime.strptime(job[19], "%Y-%m-%d").date() >= datetime.now().date()])
        st.metric("有效职位", active_jobs)
    with col3:
        expired_jobs = len(jobs) - active_jobs
        st.metric("过期职位", expired_jobs)
    with col4:
        avg_salary = sum((job[14] + job[15]) / 2 for job in jobs) / len(jobs)
        st.metric("平均薪资", f"{avg_salary:,.0f}")

    # 行业分布
    st.subheader("🏭 行业分布")
    industry_counts = {}
    for job in jobs:
        industry = job[7]
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        st.write(f"• **{industry}:** {count} 个职位 ({count/len(jobs)*100:.1f}%)")

    # 地点分布
    st.subheader("📍 工作地点分布")
    location_counts = {}
    for job in jobs:
        location = job[8]
        location_counts[location] = location_counts.get(location, 0) + 1

    for location, count in location_counts.items():
        st.write(f"• **{location}:** {count} 个职位")

    # 经验要求分布
    st.subheader("🎯 经验要求分布")
    experience_counts = {}
    for job in jobs:
        experience = job[12]
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for experience, count in experience_counts.items():
        st.write(f"• **{experience}:** {count} 个职位")

def recruitment_match_dashboard():
    """招聘匹配仪表板"""
    st.title("🎯 Recruitment Match Portal")

    # 快速统计
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("有效职位", len(jobs) if jobs else 0)
    with col2:
        st.metric("求职者", len(seekers) if seekers else 0)
    with col3:
        st.metric("匹配就绪", "✅" if jobs and seekers else "❌")

    # 页面选择
    page_option = st.sidebar.radio(
        "选择功能",
        ["智能人才匹配", "匹配统计", "使用说明"]
    )

    if page_option == "智能人才匹配":
        recruitment_match_page()
    elif page_option == "匹配统计":
        show_match_statistics()
    else:
        show_instructions()

def recruitment_match_page():
    """招聘匹配页面"""
    st.title("🎯 Recruitment Match - 智能人才匹配")

    # 获取数据
    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    if not jobs:
        st.warning("❌ 没有可用的职位信息，请先在猎头模块发布职位")
        return

    if not seekers:
        st.warning("❌ 没有可用的求职者信息，请先在Job Seeker页面填写信息")
        return

    st.success(f"📊 系统中有 {len(jobs)} 个有效职位和 {len(seekers)} 个求职者")

    # 选择职位进行匹配
    st.subheader("🔍 选择要匹配的职位")

    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("选择职位", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # 显示职位详情
    with st.expander("📋 职位详情", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**职位ID:** #{selected_job[0]}")
            st.write(f"**公司:** {selected_job[5]}")
            st.write(f"**行业:** {selected_job[6]}")
            st.write(f"**经验要求:** {selected_job[11]}")
        with col2:
            st.write(f"**地点:** {selected_job[7]}")
            st.write(f"**薪资:** {selected_job[13]:,}-{selected_job[14]:,} {selected_job[15]}")
            st.write(f"**技能要求:** {selected_job[4][:100]}...")

    # 匹配选项
    st.subheader("⚙️ 匹配设置")
    col1, col2 = st.columns(2)
    with col1:
        min_match_score = st.slider("最低匹配分数", 0, 100, 60)
    with col2:
        max_candidates = st.slider("显示前N个候选人", 1, 20, 10)

    # 执行匹配
    if st.button("🚀 开始智能匹配", type="primary", use_container_width=True):
        st.subheader("📈 匹配结果")

        progress_bar = st.progress(0)
        results = []

        for i, seeker in enumerate(seekers[:max_candidates]):
            progress = (i + 1) / min(len(seekers), max_candidates)
            progress_bar.progress(progress)

            # 使用简化匹配算法
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

        # 显示结果
        if results:
            results.sort(key=lambda x: x['match_score'], reverse=True)
            st.success(f"🎉 找到 {len(results)} 个匹配的候选人 (分数 ≥ {min_match_score})")

            for i, result in enumerate(results):
                score_color = "🟢" if result['match_score'] >= 80 else "🟡" if result['match_score'] >= 60 else "🔴"

                with st.expander(f"{score_color} #{i+1} {result['name']} - {result['match_score']}分", expanded=i < 2):
                    col1, col2 = st.columns(2)

                    with col1:
                        st.write("**候选人信息:**")
                        st.write(f"**ID:** #{result['seeker_id']}")
                        st.write(f"**教育背景:** {result['education']}")
                        st.write(f"**工作经验:** {result['experience']}")
                        st.write(f"**当前背景:** {result['current_title']}")
                        st.write(f"**技能:** {result['raw_data'][2][:100]}...")

                    with col2:
                        st.write("**匹配分析:**")
                        st.write(f"**匹配分数:** {score_color} {result['match_score']}分")
                        st.write(f"**薪资匹配:** {result['analysis'].get('salary_match', '一般')}")
                        st.write(f"**文化契合:** {result['analysis'].get('culture_fit', '中')}")

                        if 'key_strengths' in result['analysis']:
                            st.write("**核心优势:**")
                            for strength in result['analysis']['key_strengths']:
                                st.write(f"✅ {strength}")

                        if 'potential_gaps' in result['analysis']:
                            st.write("**关注点:**")
                            for gap in result['analysis']['potential_gaps']:
                                st.write(f"⚠️ {gap}")

                    if 'recommendation' in result['analysis']:
                        st.info(f"**推荐建议:** {result['analysis']['recommendation']}")

                    # 操作按钮
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        if st.button("📞 联系候选人", key=f"contact_{result['seeker_id']}"):
                            st.success(f"已标记联系: {result['name']}")
                    with col_btn2:
                        if st.button("💼 安排面试", key=f"interview_{result['seeker_id']}"):
                            st.success(f"已安排面试: {result['name']}")
        else:
            st.warning("😔 没有找到匹配的候选人，请调整匹配条件")

def ai_interview_dashboard():
    """AI面试仪表板"""
    st.title("🤖 AI模拟面试系统")

    # 快速统计
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("可用职位", len(jobs) if jobs else 0)
    with col2:
        st.metric("个人资料", "✅" if seeker_profile else "❌")
    with col3:
        if 'interview' in st.session_state:
            progress = st.session_state.interview['current_question']
            total = st.session_state.interview['total_questions']
            st.metric("面试进度", f"{progress}/{total}")
        else:
            st.metric("面试状态", "待开始")

    # 页面选择
    page_option = st.sidebar.radio(
        "选择功能",
        ["开始模拟面试", "面试准备指导", "使用说明"]
    )

    if page_option == "开始模拟面试":
        ai_interview_page()
    elif page_option == "面试准备指导":
        show_interview_guidance()
    else:
        show_interview_instructions()

def show_interview_guidance():
    """显示面试准备指导"""
    st.header("🎯 面试准备指导")

    st.info("""
    **面试准备建议:**

    ### 📚 技术面试准备
    1. **复习核心技能**: 确保掌握职位要求的关键技术
    2. **准备项目案例**: 准备2-3个能展示您能力的项目
    3. **练习编码题**: 针对技术职位准备算法和数据结构

    ### 💼 行为面试准备
    1. **STAR法则**:  Situation-Task-Action-Result
    2. **准备成功案例**: 展示您如何解决问题和创造价值
    3. **了解公司文化**: 研究公司的价值观和工作方式

    ### 🎯 沟通技巧
    1. **清晰表达**: 结构化您的回答
    2. **积极倾听**: 确保理解问题的核心
    3. **展示热情**: 表达对职位和公司的兴趣
    """)

def show_interview_instructions():
    """显示使用说明"""
    st.header("📖 AI模拟面试使用说明")

    st.info("""
    **AI模拟面试功能指南:**

    ### 🚀 开始面试
    1. **选择职位**: 从猎头发布的职位中选择一个进行模拟面试
    2. **开始面试**: AI会根据职位要求生成相关问题
    3. **回答问题**: 针对每个问题提供详细的回答

    ### 📊 面试流程
    - **10个问题**: 包含技术、行为、情景等多种类型
    - **实时评估**: AI会评估每个回答的质量
    - **个性化问题**: 后续问题基于您之前的回答

    ### 🎯 获得反馈
    - **详细评分**: 每个问题的具体评分和反馈
    - **总体评价**: 完整的面试表现总结
    - **改进建议**: 针对性的职业发展建议

    **提示**: 请确保在网络稳定的环境下使用，以便AI能正常生成问题和评估回答。
    """)

# 在侧边栏添加调试工具
with st.sidebar:
    st.markdown("---")
    st.subheader("🔧 数据库调试")
    
    if st.button("查看所有求职者记录"):
        try:
            conn = sqlite3.connect('job_seeker.db')
            c = conn.cursor()
            c.execute("SELECT job_seeker_id, timestamp, education_level, primary_role FROM job_seekers ORDER BY id DESC")
            results = c.fetchall()
            conn.close()
            
            if results:
                st.write("📋 所有求职者记录:")
                for record in results:
                    st.write(f"- ID: {record[0]}, 时间: {record[1]}, 学历: {record[2]}, 角色: {record[3]}")
            else:
                st.write("暂无求职者记录")
        except Exception as e:
            st.error(f"查询失败: {e}")
    
    # 显示当前session状态
    current_id = st.session_state.get('job_seeker_id')
    if current_id:
        st.info(f"当前Session ID: **{current_id}**")

# 侧边栏导航
st.sidebar.title("🔍 导航")

# 导航按钮
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

# 页面路由
if st.session_state.current_page == "main":
    main_analyzer_page()
elif st.session_state.current_page == "job_recommendations":
    job_seeker_id = st.session_state.get('job_seeker_id')

    # 检查是否有保存的求职者数据
    if not job_seeker_id:
        st.warning("⚠️ 请先在 Job Seeker 页面保存您的个人信息")
        st.info("👉 切换到 'Job Seeker' 页面填写并保存您的资料")
        
        # 提供快捷跳转
        if st.button("前往 Job Seeker 页面"):
            st.session_state.current_page = "main"
            st.rerun()
    else:
        # 调用工作推荐页面函数
        job_recommendations_page(job_seeker_id)

elif st.session_state.current_page == "head_hunter":
    enhanced_head_hunter_page()
elif st.session_state.current_page == "recruitment_match":
    recruitment_match_dashboard()
elif st.session_state.current_page == "ai_interview":
    ai_interview_dashboard()


# 侧边栏信息
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 💡 使用说明

1. **主页**: 智能简历-JD匹配分析器
2. **Job Seeker**: 填写信息 → 自动推荐职位
3. **Job Match**: 查看AI匹配的职位
4. **Head Hunter**: 发布和管理招聘职位
5. **Recruitment Match**: 智能匹配候选人与职位
6. **AI Interview**: 模拟面试和技能评估
7. **DB Verify**: 验证数据存储
""")
                    
# Footer
st.markdown("---")
st.caption("🤖 Powered by GPT-4, Pinecone Vector Search, and RapidAPI LinkedIn Jobs")

# 应用启动
if __name__ == "__main__":
    # 确保应用正常运行
    pass