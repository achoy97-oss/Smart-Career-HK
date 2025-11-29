import sqlite3
import streamlit as st
from datetime import datetime
import uuid

class JobSeekerDB:
    def __init__(self, db_path='job_seeker.db'):
        self.db_path = db_path

    def _connect(self):
        """Internal method: establish database connection"""
        return sqlite3.connect(self.db_path)

    def get_job_seeker_by_id(self, job_seeker_id):
        """Get job seeker data by job_seeker_id"""
        try:
            conn = self._connect()
            c = conn.cursor()

            # Get column list
            c.execute("PRAGMA table_info(job_seekers)")
            columns = [col[1] for col in c.fetchall()]

            # Query
            c.execute("SELECT * FROM job_seekers WHERE job_seeker_id = ?", (job_seeker_id,))
            result = c.fetchone()

            conn.close()

            if result:
                return dict(zip(columns, result))
            else:
                print(f"❌ No job seeker found with ID {job_seeker_id}")
                return None

        except Exception as e:
            print(f"❌ Error getting job seeker data: {e}")
            return None
        
    def get_latest_job_seeker_id(self):
        """Get the latest job_seeker_id"""
        try:
            conn = self._connect()
            c = conn.cursor()

            c.execute("SELECT job_seeker_id FROM job_seekers ORDER BY id DESC LIMIT 1")
            result = c.fetchone()

            conn.close()
            return result[0] if result else None

        except Exception as e:
            print(f"❌ Error getting latest job seeker ID: {e}")
            return None

    def get_latest_job_seeker_data(self):
        """Get complete data of the latest job seeker"""
        try:
            conn = self._connect()
            c = conn.cursor()

            # Get column list
            c.execute("PRAGMA table_info(job_seekers)")
            columns = [col[1] for col in c.fetchall()]

            c.execute("SELECT * FROM job_seekers ORDER BY id DESC LIMIT 1")
            result = c.fetchone()

            conn.close()

            return dict(zip(columns, result)) if result else None

        except Exception as e:
            print(f"❌ Error getting latest job seeker data: {e}")
            return None


# Functions moved to the bottom
def init_database():
    """Initialize job seeker database"""
    try:
        conn = sqlite3.connect('job_seeker.db')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS job_seekers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_seeker_id TEXT UNIQUE,
                timestamp TEXT,
                education_level TEXT,
                major TEXT,
                graduation_status TEXT,
                university_background TEXT,
                languages TEXT,
                certificates TEXT,
                hard_skills TEXT,
                soft_skills TEXT,
                work_experience TEXT,
                project_experience TEXT,
                location_preference TEXT,
                industry_preference TEXT,
                salary_expectation TEXT,
                benefits_expectation TEXT,
                primary_role TEXT,
                simple_search_terms TEXT
            )
        """)
        conn.commit()
        conn.close()
        print("✅ Job seeker database initialized successfully")
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")

def generate_job_seeker_id():
    """Generate unique job seeker ID"""
    return f"JS_{uuid.uuid4().hex[:8].upper()}"

def save_job_seeker_info(education_level, major, graduation_status, university_background,
                        languages, certificates, hard_skills, soft_skills, work_experience,
                        project_experience, location_preference, industry_preference, 
                        salary_expectation, benefits_expectation, primary_role="", simple_search_terms=""):
    """Save job seeker information to database and return job_seeker_id"""
    try:
        # Ensure database is initialized
        init_database()
        
        conn = sqlite3.connect('job_seeker.db')
        c = conn.cursor()

        # Generate unique job_seeker_id
        job_seeker_id = generate_job_seeker_id()

        c.execute("""
            INSERT INTO job_seekers (
                job_seeker_id, timestamp, education_level, major, graduation_status, university_background,
                languages, certificates, hard_skills, soft_skills, work_experience, 
                project_experience, location_preference, industry_preference,
                salary_expectation, benefits_expectation, primary_role, simple_search_terms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            job_seeker_id,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            education_level, major, graduation_status, university_background,
            languages, certificates, hard_skills, soft_skills, work_experience,
            project_experience, location_preference, industry_preference, 
            salary_expectation, benefits_expectation, primary_role, simple_search_terms
        ))

        conn.commit()
        conn.close()
        
        print(f"✅ Data saved successfully, Job Seeker ID: {job_seeker_id}")
        return job_seeker_id
        
    except sqlite3.Error as e:
        print(f"❌ SQLite error: {e}")
        return None
    except Exception as e:
        print(f"❌ General error saving job seeker info: {e}")
        return None



def init_head_hunter_database():
    """Initialize headhunter positions database"""
    try:
        conn = sqlite3.connect('head_hunter_jobs.db')
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS head_hunter_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                job_title TEXT,
                job_description TEXT,
                main_responsibilities TEXT,
                required_skills TEXT,
                client_company TEXT,
                industry TEXT,
                work_location TEXT,
                work_type TEXT,
                company_size TEXT,
                employment_type TEXT,
                experience_level TEXT,
                visa_support TEXT,
                min_salary REAL,
                max_salary REAL,
                currency TEXT,
                benefits TEXT,
                application_method TEXT,
                job_valid_until TEXT
            )
        """)
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Database initialization failed: {e}")

def save_head_hunter_job(job_data):
    """Save headhunter position information to database"""
    try:
        conn = sqlite3.connect('head_hunter_jobs.db')
        c = conn.cursor()

        c.execute("""
            INSERT INTO head_hunter_jobs (
                timestamp, job_title, job_description, main_responsibilities, required_skills,
                client_company, industry, work_location, work_type, company_size,
                employment_type, experience_level, visa_support,
                min_salary, max_salary, currency, benefits, application_method, job_valid_until
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            job_data['job_title'],
            job_data['job_description'],
            job_data['main_responsibilities'],
            job_data['required_skills'],
            job_data['client_company'],
            job_data['industry'],
            job_data['work_location'],
            job_data['work_type'],
            job_data['company_size'],
            job_data['employment_type'],
            job_data['experience_level'],
            job_data['visa_support'],
            job_data['min_salary'],
            job_data['max_salary'],
            job_data['currency'],
            job_data['benefits'],
            job_data['application_method'],
            job_data['job_valid_until']
        ))

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        st.error(f"Failed to save position: {e}")
        return False
    
class HeadhunterDB:
    def get_all_head_hunter_jobs(self):
        try:
            conn = sqlite3.connect('head_hunter_jobs.db')
            c = conn.cursor()
            c.execute("SELECT * FROM head_hunter_jobs ORDER BY id DESC")
            data = c.fetchall()   # ← FIXED
            conn.close()
            return data
        except Exception as e:
            st.error(f"Failed to get job positions: {e}")
            return []


conn = sqlite3.connect("head_hunter_jobs.db")
c = conn.cursor()

def get_job_seeker_search_fields(job_seeker_id: str):
    conn = sqlite3.connect("job_seeker.db")
    c = conn.cursor()

    c.execute("""
        SELECT 
            education_level, major, graduation_status, university_background,
            languages, certificates, hard_skills, soft_skills,
            work_experience, project_experience, location_preference,
            industry_preference, salary_expectation, benefits_expectation,
            primary_role, simple_search_terms
        FROM job_seekers
        WHERE job_seeker_id = ?
    """, (job_seeker_id,))

    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "education_level": row[0] or "",
        "major": row[1] or "",
        "graduation_status": row[2] or "",
        "university_background": row[3] or "",
        "languages": row[4] or "",
        "certificates": row[5] or "",
        "hard_skills": row[6] or "",
        "soft_skills": row[7] or "",
        "work_experience": row[8] or "",
        "project_experience": row[9] or "",
        "location_preference": row[10] or "",
        "industry_preference": row[11] or "",
        "salary_expectation": row[12] or "",
        "benefits_expectation": row[13] or "",
        "primary_role": row[14] or "",
        "simple_search_terms": row[15] or "",
    }