"""
Job Matcher Backend - COMPLETE VERSION
With improved error handling and simplified RapidAPI queries
"""

import os
import re
import time
import json
import docx
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from docx import Document
import PyPDF2
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import pandas as pd
import openai
from openai import AzureOpenAI
from config import Config
import streamlit as st
import sqlite3

# Initialize config
Config.setup()


# ============================================================================
# RESUME PARSER - NO HARDCODED SKILLS
# ============================================================================

class ResumeParser:
    """Parse resume from PDF or DOCX - Let GPT-4 extract skills"""
    
    def __init__(self):
        pass
    
    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file object"""
        try:
            text = ""
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error reading PDF: {str(e)}")
    
    def extract_text_from_docx(self, docx_file) -> str:
        """Extract text from DOCX file object"""
        try:
            doc = Document(docx_file)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
            return text
        except Exception as e:
            raise Exception(f"Error reading DOCX: {str(e)}")
    
    def extract_text(self, file_obj, filename: str) -> str:
        """Extract text from uploaded file"""
        if filename.lower().endswith('.pdf'):
            return self.extract_text_from_pdf(file_obj)
        elif filename.lower().endswith('.docx'):
            return self.extract_text_from_docx(file_obj)
        else:
            raise ValueError("Unsupported file format. Use PDF or DOCX.")
    
    def parse_resume(self, file_obj, filename: str) -> Dict:
        """Parse resume and extract raw text only"""
        try:
            text = self.extract_text(file_obj, filename)
            
            if not text or len(text.strip()) < 50:
                raise ValueError("Could not extract sufficient text from resume")
            
            resume_data = {
                'raw_text': text,
                'text_length': len(text),
                'word_count': len(text.split()),
                'filename': filename
            }
            
            return resume_data
            
        except Exception as e:
            raise Exception(f"Error parsing resume: {str(e)}")


# ============================================================================
# GPT-4 JOB ROLE DETECTOR - EXTRACTS SKILLS DYNAMICALLY
# ============================================================================

class GPT4JobRoleDetector:
    """Use GPT-4 to detect job roles AND extract skills dynamically"""
    
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=Config.AZURE_ENDPOINT,
            api_key=Config.AZURE_API_KEY,
            api_version=Config.AZURE_API_VERSION
        )
        self.model = Config.AZURE_MODEL
    
    def analyze_resume_for_job_roles(self, resume_data: Dict) -> Dict:
        """Analyze resume with GPT-4 - Extract ALL skills dynamically"""
        
        resume_text = resume_data.get('raw_text', '')[:3000]
        
        system_prompt = """You are an expert career advisor and resume analyst.

Analyze the resume and extract:
1. ALL skills (technical, soft skills, tools, languages, frameworks, methodologies, domain knowledge)
2. Job role recommendations
3. Seniority level
4. SIMPLE job search keywords (for job board APIs)

IMPORTANT for job search:
- Provide a SIMPLE primary role (e.g., "Program Manager" not complex OR/AND queries)
- Keep search keywords SHORT and COMMON
- Avoid complex boolean logic in search queries

Return JSON with this EXACT structure:
{
    "primary_role": "Simple job title (e.g., Program Manager)",
    "simple_search_terms": ["term1", "term2", "term3"],
    "confidence": 0.95,
    "seniority_level": "Junior/Mid-Level/Senior/Lead/Executive",
    "skills": ["skill1", "skill2", "skill3", ...],
    "core_strengths": ["strength1", "strength2", "strength3"],
    "job_search_keywords": ["keyword1", "keyword2"],
    "optimal_search_query": "Simple search string (just the job title)",
    "location_preference": "Detected or 'United States'",
    "industries": ["industry1", "industry2"],
    "alternative_roles": ["role1", "role2", "role3"]
}"""

        user_prompt = f"""Analyze this resume and extract ALL information:

RESUME:
{resume_text}

IMPORTANT - Extract ALL skills including:
- Programming languages (Python, R, SQL, etc.)
- Tools and software (Tableau, Salesforce, Excel, etc.)
- Methodologies (Agile, Scrum, Kanban, etc.)
- Soft skills (Leadership, Communication, etc.)
- Domain expertise (Banking, Finance, Analytics, etc.)
- Technical skills (Data Analysis, Machine Learning, etc.)
- Languages (English, Cantonese, Mandarin, etc.)

For job search, provide SIMPLE terms that would work on LinkedIn/Indeed (not complex boolean queries).

Be thorough and creative!"""

        try:
            print("🤖 Calling GPT-4 for resume analysis...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_analysis = json.loads(response.choices[0].message.content)
            print(f"✅ GPT-4 analysis complete! Found {len(ai_analysis.get('skills', []))} skills")
            return ai_analysis
            
        except Exception as e:
            print(f"❌ GPT-4 Error: {e}")
            return self._fallback_analysis()
    
    def _fallback_analysis(self) -> Dict:
        """Fallback if GPT-4 fails"""
        return {
            "primary_role": "Professional",
            "simple_search_terms": ["Professional"],
            "confidence": 0.5,
            "seniority_level": "Mid-Level",
            "skills": ["General Skills"],
            "core_strengths": ["Adaptable", "Professional"],
            "job_search_keywords": ["Professional"],
            "optimal_search_query": "Professional",
            "location_preference": "United States",
            "industries": ["General"],
            "alternative_roles": ["Specialist", "Consultant"]
        }


# ============================================================================
# LINKEDIN JOB SEARCHER - WITH BETTER ERROR HANDLING
# ============================================================================

class LinkedInJobSearcher:
    """Search for jobs using RapidAPI LinkedIn API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://linkedin-job-search-api.p.rapidapi.com/active-jb-7d"
        self.headers = {
            "x-rapidapi-key": api_key,
            "x-rapidapi-host": "linkedin-job-search-api.p.rapidapi.com"
        }
    
    def test_api_connection(self) -> Tuple[bool, str]:
        """Test if the API is working"""
        try:
            querystring = {
                "limit": "5",
                "offset": "0",
                "title_filter": "\"Engineer\"",
                "location_filter": "\"Hong Kong\"",
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=10
            )
            
            if response.status_code == 200:
                return True, "API is working"
            elif response.status_code == 403:
                return False, "API key is invalid or expired (403 Forbidden)"
            elif response.status_code == 429:
                return False, "Rate limit exceeded (429 Too Many Requests)"
            else:
                return False, f"API returned status code {response.status_code}"
        
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def search_jobs(
        self,
        keywords: str,
        location: str = "Hong Kong",
        limit: int = 20
    ) -> List[Dict]:
        """Search LinkedIn jobs with simplified queries"""
        
        # Simplify complex queries
        simple_keywords = self._simplify_query(keywords)
        
        querystring = {
            "limit": str(limit),
            "offset": "0",
            "title_filter": f'"{simple_keywords}"',
            "location_filter": f'"{location}"',
            "description_type": "text"
        }
        
        try:
            print(f"🔍 Searching RapidAPI...")
            print(f"   Original query: {keywords}")
            print(f"   Simplified to: {simple_keywords}")
            print(f"   Location: {location}")
            
            response = requests.get(
                self.base_url, 
                headers=self.headers, 
                params=querystring, 
                timeout=30
            )
            
            print(f"📊 API Response Status: {response.status_code}")
            
            if response.status_code == 403:
                print("❌ API Key Error: 403 Forbidden")
                print("   Your RapidAPI key might be invalid or expired")
                print("   Check: https://rapidapi.com/")
                return []
            
            elif response.status_code == 429:
                print("❌ Rate Limit: 429 Too Many Requests")
                print("   Wait a few minutes or upgrade your RapidAPI plan")
                return []
            
            elif response.status_code != 200:
                print(f"❌ API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}")
                return []
            
            data = response.json()
            
            # Handle different response formats
            if isinstance(data, list):
                jobs = data
            elif isinstance(data, dict):
                jobs = data.get('data', data.get('jobs', data.get('results', [])))
            else:
                jobs = []
            
            if not jobs:
                print(f"⚠️ No jobs found for '{simple_keywords}'")
                print("   Trying fallback searches...")
                
                # Try alternative searches
                for alternative in self._get_alternative_searches(simple_keywords):
                    alt_jobs = self._try_alternative_search(alternative, location, 10)
                    if alt_jobs:
                        print(f"✅ Found {len(alt_jobs)} jobs with alternative search: {alternative}")
                        jobs.extend(alt_jobs)
                        if len(jobs) >= 10:
                            break
            
            normalized = self._normalize_jobs(jobs)
            print(f"✅ Retrieved {len(normalized)} jobs from RapidAPI")
            return normalized
            
        except Exception as e:
            print(f"❌ LinkedIn API Error: {str(e)}")
            return []
    
    def _simplify_query(self, query: str) -> str:
        """Simplify complex boolean queries to simple terms"""
        # Remove boolean operators and parentheses
        simple = query.replace(" OR ", " ").replace(" AND ", " ")
        simple = simple.replace("(", "").replace(")", "")
        simple = simple.replace('"', "")
        
        # Take first few words (most important)
        words = simple.split()[:3]
        return " ".join(words)
    
    def _get_alternative_searches(self, primary_query: str) -> List[str]:
        """Generate alternative search terms"""
        alternatives = [
            primary_query.split()[0] if primary_query.split() else primary_query,  # First word only
            "Manager",  # Generic fallback
            "Analyst",  # Generic fallback
        ]
        return alternatives
    
    def _try_alternative_search(self, keywords: str, location: str, limit: int) -> List[Dict]:
        """Try an alternative search"""
        try:
            querystring = {
                "limit": str(limit),
                "offset": "0",
                "title_filter": f'"{keywords}"',
                "location_filter": f'"{location}"',
                "description_type": "text"
            }
            
            response = requests.get(
                self.base_url,
                headers=self.headers,
                params=querystring,
                timeout=20
            )
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    return data
                elif isinstance(data, dict):
                    return data.get('data', data.get('jobs', data.get('results', [])))
            
            return []
        
        except:
            return []
    
    def _normalize_jobs(self, jobs: List[Dict]) -> List[Dict]:
        """Normalize job structure"""
        normalized_jobs = []
        
        for job in jobs:
            try:
                # Handle location
                location = "Remote"
                if job.get('locations_derived') and len(job['locations_derived']) > 0:
                    location = job['locations_derived'][0]
                elif job.get('locations_raw'):
                    try:
                        loc_raw = job['locations_raw'][0]
                        if isinstance(loc_raw, dict) and 'address' in loc_raw:
                            addr = loc_raw['address']
                            city = addr.get('addressLocality', '')
                            region = addr.get('addressRegion', '')
                            if city and region:
                                location = f"{city}, {region}"
                    except:
                        pass
                
                normalized_job = {
                    'id': job.get('id', f"job_{len(normalized_jobs)}"),
                    'title': job.get('title', 'Unknown Title'),
                    'company': job.get('organization', 'Unknown Company'),
                    'location': location,
                    'description': job.get('description_text', ''),
                    'url': job.get('url', ''),
                    'posted_date': job.get('date_posted', 'Unknown'),
                }
                
                normalized_jobs.append(normalized_job)
                
            except Exception as e:
                continue
        
        return normalized_jobs


# ============================================================================
# JOB MATCHER - PINECONE SEMANTIC SEARCH & RANKING
# ============================================================================

class JobMatcher:
    """Match resume to jobs using Pinecone semantic search and skill matching"""
    
    def __init__(self):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=Config.PINECONE_API_KEY)
        
        # Initialize embedding model
        print("📦 Loading sentence transformer model...")
        self.model = SentenceTransformer(Config.MODEL_NAME)
        print("✅ Model loaded!")
        
        # Create/connect to index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        existing_indexes = self.pc.list_indexes()
        index_names = [idx['name'] for idx in existing_indexes]
        
        if Config.INDEX_NAME not in index_names:
            print(f"🔨 Creating new Pinecone index: {Config.INDEX_NAME}")
            self.pc.create_index(
                name=Config.INDEX_NAME,
                dimension=Config.EMBEDDING_DIMENSION,
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region=Config.PINECONE_ENVIRONMENT
                )
            )
            time.sleep(2)
        else:
            print(f"✅ Using existing Pinecone index: {Config.INDEX_NAME}")
        
        self.index = self.pc.Index(Config.INDEX_NAME)
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding vector"""
        text = str(text).strip()
        if not text:
            text = "empty"
        
        embedding = self.model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    
    def index_jobs(self, jobs: List[Dict]) -> int:
        """Index jobs in Pinecone"""
        if not jobs:
            return 0
        
        vectors_to_upsert = []
        
        for job in jobs:
            try:
                job_text = f"{job['title']} {job['company']} {job['description']}"
                embedding = self.generate_embedding(job_text)
                
                vectors_to_upsert.append({
                    'id': job['id'],
                    'values': embedding,
                    'metadata': {
                        'title': job['title'][:512],
                        'company': job['company'][:512],
                        'location': job['location'][:512],
                        'description': job['description'][:1000],
                        'url': job.get('url', '')[:512],
                        'posted_date': str(job.get('posted_date', ''))[:100]
                    }
                })
                
            except Exception as e:
                print(f"⚠️ Error indexing job {job.get('id', 'unknown')}: {e}")
                continue
        
        if vectors_to_upsert:
            self.index.upsert(vectors=vectors_to_upsert)
            return len(vectors_to_upsert)
        
        return 0
    
    def search_similar_jobs(self, resume_data: Dict, ai_analysis: Dict, top_k: int = 20) -> List[Dict]:
        """Search for similar jobs using semantic similarity"""
        try:
            # Create rich query from resume + AI analysis
            primary_role = ai_analysis.get('primary_role', '')
            skills = ' '.join(ai_analysis.get('skills', [])[:20])
            resume_snippet = resume_data.get('raw_text', '')[:1000]
            
            query_text = f"{primary_role} {skills} {resume_snippet}"
            
            print(f"🎯 Creating semantic embedding for resume...")
            query_embedding = self.generate_embedding(query_text)
            
            print(f"🔍 Searching Pinecone for top {top_k} matches...")
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            matched_jobs = []
            for match in results['matches']:
                job = {
                    'id': match['id'],
                    'similarity_score': float(match['score']) * 100,
                    **match['metadata']
                }
                matched_jobs.append(job)
            
            print(f"✅ Found {len(matched_jobs)} semantic matches")
            return matched_jobs
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


# ============================================================================
# MAIN BACKEND - ORCHESTRATES EVERYTHING
# ============================================================================

class JobSeekerBackend:
    """Main backend with FULL integration"""
    
    def __init__(self):
        print("🚀 Initializing Job Matcher Backend...")
        Config.validate()
        self.resume_parser = ResumeParser()
        self.gpt4_detector = GPT4JobRoleDetector()
        self.job_searcher = LinkedInJobSearcher(Config.RAPIDAPI_KEY)
        self.matcher = JobMatcher()
        
        # Test API connection
        print("\n🧪 Testing RapidAPI connection...")
        is_working, message = self.job_searcher.test_api_connection()
        if is_working:
            print(f"✅ {message}")
        else:
            print(f"⚠️ WARNING: {message}")
            print("   Job search may not work properly!")
        
        print("\n✅ Backend initialized!\n")
    
    def process_resume(self, file_obj, filename: str) -> Tuple[Dict, Dict]:
        """Process resume and get AI analysis"""
        print(f"📄 Processing resume: {filename}")
        
        # Parse resume
        resume_data = self.resume_parser.parse_resume(file_obj, filename)
        print(f"✅ Extracted {resume_data['word_count']} words from resume")
        
        # Get GPT-4 analysis
        ai_analysis = self.gpt4_detector.analyze_resume_for_job_roles(resume_data)
        
        # Add skills to resume_data
        resume_data['skills'] = ai_analysis.get('skills', [])
        
        return resume_data, ai_analysis
    
    def search_and_match_jobs(self, resume_data: Dict, ai_analysis: Dict, num_jobs: int = 30) -> List[Dict]:
        """Search for jobs GLOBALLY and rank by match quality"""
        
        # Use simplified search query
        search_query = ai_analysis.get('primary_role', 'Professional')
        location = "United States"
        
        print(f"\n{'='*60}")
        print(f"🌍 SEARCHING JOBS GLOBALLY")
        print(f"{'='*60}")
        print(f"🔍 Search Query: {search_query}")
        print(f"📍 Location: {location}")
        print(f"{'='*60}\n")
        
        # Search jobs
        jobs = self.job_searcher.search_jobs(
            keywords=search_query,
            location=location,
            limit=num_jobs
        )
        
        if not jobs or len(jobs) == 0:
            print("\n❌ No jobs found from RapidAPI")
            print("\n💡 Possible reasons:")
            print("   - API key might be invalid/expired")
            print("   - Rate limit exceeded")
            print("   - No jobs available for this search term")
            print("\n🔧 Suggestions:")
            print("   - Check your RapidAPI account at https://rapidapi.com/")
            print("   - Wait a few minutes if rate limited")
            print("   - Try with a different resume/role")
            return []
        
        print(f"\n✅ Retrieved {len(jobs)} jobs from RapidAPI")
        print(f"📊 Indexing jobs in Pinecone...")
        
        # Index jobs
        indexed = self.matcher.index_jobs(jobs)
        print(f"✅ Indexed {indexed} jobs in vector database")
        
        # Wait for indexing
        print("⏳ Waiting for indexing to complete...")
        time.sleep(2)
        
        # Match resume to jobs
        print(f"\n🎯 MATCHING & RANKING JOBS")
        print(f"{'='*60}")
        matched_jobs = self.matcher.search_similar_jobs(
            resume_data, 
            ai_analysis, 
            top_k=min(20, len(jobs))
        )
        
        if not matched_jobs:
            print("⚠️ No matches found")
            return []
        
        # Calculate match scores
        matched_jobs = self._calculate_match_scores(matched_jobs, ai_analysis)
        
        # Sort by combined score
        matched_jobs.sort(key=lambda x: x.get('combined_score', 0), reverse=True)
        
        print(f"✅ Ranked {len(matched_jobs)} jobs by match quality")
        print(f"{'='*60}\n")
        
        return matched_jobs
    
    def _calculate_match_scores(self, jobs: List[Dict], ai_analysis: Dict) -> List[Dict]:
        """Calculate detailed match scores - 60% semantic + 40% skill match"""
        
        candidate_skills = set([s.lower() for s in ai_analysis.get('skills', [])])
        
        print(f"📊 Calculating match scores using {len(candidate_skills)} candidate skills...")
        
        for job in jobs:
            description = job.get('description', '').lower()
            title = job.get('title', '').lower()
            
            # Count skill matches
            matched_skills = []
            for skill in candidate_skills:
                if skill in description or skill in title:
                    matched_skills.append(skill)
            
            # Calculate skill match percentage
            skill_match_pct = (len(matched_skills) / len(candidate_skills) * 100) if candidate_skills else 0
            
            # Semantic similarity (from Pinecone)
            semantic_score = job.get('similarity_score', 0)
            
            # Combined score: 60% semantic + 40% skill match
            combined_score = (0.6 * semantic_score) + (0.4 * skill_match_pct)
            
            # Add to job
            job['skill_match_percentage'] = round(skill_match_pct, 1)
            job['matched_skills'] = list(matched_skills)[:10]
            job['matched_skills_count'] = len(matched_skills)
            job['combined_score'] = round(combined_score, 1)
            job['semantic_score'] = round(semantic_score, 1)
        
        return jobs
    
    @staticmethod
    def parse_cv_with_ai(cv_text):
        prompt = f"""
Below is the cv text of a candidate. 
Please extract structured information (leave blank if missing):
cv_text: '''{cv_text}'''

Please output JSON, fields including:
- education_level(doctor/master/bachelor/associate/highschool)
- major
- graduation_status(fresh graduate/experienced/in study)
- university_background(985 university/211 university/overseas university/regular university/other)
- languages
- certificates
- hard_skills
- soft_skills
- work_experience(fresh graduate/1-3 years/3-5 years/5-10 years/10+ years)
- project_experience
- location_preference
- industry_preference
- salary_expectation
- benefits_expectation

Please return the result in the JSON format only, no extra explanation.
"""

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        try:
            return json.loads(response.choices[0].message.content)
        except Exception:
            return {}

class JobMatcherBackend:
    """Main backend with FULL integration"""
    
    def fetch_real_jobs(self, search_query, location="", country="us", num_pages=1):
        """Get actual job data from JSearch API"""
        try:
            # JSearch API configuration
            API_KEY = "your_jsearch_api_key_here"  # You need to get api key from https://jsearch.app/
            BASE_URL = "https://jsearch.p.rapidapi.com/search"
            
            headers = {
                "X-RapidAPI-Key": API_KEY,
                "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
            }
            
            all_jobs = []
            
            for page in range(1, num_pages + 1):
                querystring = {
                    "query": f"{search_query} {location}",
                    "page": str(page),
                    "num_pages": "1"
                }
                
                response = requests.get(BASE_URL, headers=headers, params=querystring)
                
                if response.status_code == 200:
                    data = response.json()
                    jobs = data.get('data', [])
                    all_jobs.extend(jobs)
                    print(f"✅ Page {page} fetched {len(jobs)} jobs")
                else:
                    print(f"❌ API request failed: {response.status_code}")
                    break
            
            print(f"🎯 Found total of {len(all_jobs)} positions")
            return all_jobs
            
        except Exception as e:
            print(f"❌ Failed to fetch jobs: {e}")
            # return simulated data as fallback
            return self.get_mock_jobs(search_query, location)

    def get_mock_jobs(self, search_query, location):
        """return mock job data (used when API is unavailable)"""
        print("🔄 Using simulated data...")
        
        mock_jobs = [
            {
                'job_title': f'Senior {search_query}',
                'employer_name': 'Tech Company Inc.',
                'job_city': location or 'Hong Kong',
                'job_country': 'HK',
                'job_employment_type': 'FULLTIME',
                'job_posted_at': '2024-01-15',
                'job_description': f'We are looking for a skilled {search_query} to join our team. Requirements include strong programming skills and experience.',
                'job_apply_link': 'https://example.com/apply/1',
                'job_highlights': {
                    'Qualifications': ['Bachelor\'s degree in Computer Science', '3+ years of experience'],
                    'Responsibilities': ['Develop software applications', 'Collaborate with team members']
                }
            },
            {
                'job_title': f'Junior {search_query}',
                'employer_name': 'Startup Solutions',
                'job_city': location or 'Hong Kong',
                'job_country': 'HK',
                'job_employment_type': 'FULLTIME',
                'job_posted_at': '2024-01-10',
                'job_description': f'Entry-level position for {search_query}. Great learning opportunity for recent graduates.',
                'job_apply_link': 'https://example.com/apply/2',
                'job_highlights': {
                    'Qualifications': ['Degree in related field', 'Basic programming knowledge'],
                    'Responsibilities': ['Assist senior developers', 'Learn new technologies']
                }
            },
            {
                'job_title': f'{search_query} Specialist',
                'employer_name': 'Global Corp',
                'job_city': location or 'Hong Kong',
                'job_country': 'HK',
                'job_employment_type': 'CONTRACTOR',
                'job_posted_at': '2024-01-08',
                'job_description': f'Contract position for {search_query} with potential for extension.',
                'job_apply_link': 'https://example.com/apply/3',
                'job_highlights': {
                    'Qualifications': ['Proven track record', 'Excellent communication skills'],
                    'Responsibilities': ['Project development', 'Client meetings']
                }
            }
        ]
        
        return mock_jobs

    def calculate_job_match_score(self, job_seeker_data, job_data):
        """calcalate job match score between job seeker and job data"""
        try:
            score = 0
            max_score = 100
            matched_skills = []
            
            # 1. Skill match (40%)
            job_seeker_skills = job_seeker_data.get('hard_skills', '').lower()
            job_description = job_data.get('job_description', '').lower()
            
            if job_seeker_skills:
                skills_list = [skill.strip().lower() for skill in job_seeker_skills.split(',')]
                for skill in skills_list:
                    if skill and skill in job_description:
                        score += 5  # Each match score add 5 points
                        matched_skills.append(skill)
                        if score >= 40:  # Max skill points at 40
                            score = 40
                            break
            
            # 2. Experience match (20%)
            job_seeker_experience = job_seeker_data.get('work_experience', '').lower()
            if 'senior' in job_data.get('job_title', '').lower() and 'senior' in job_seeker_experience.lower():
                score += 20
            elif 'junior' in job_data.get('job_title', '').lower() and 'junior' in job_seeker_experience.lower():
                score += 20
            elif 'entry' in job_data.get('job_title', '').lower() and 'fresh' in job_seeker_experience.lower():
                score += 20
            else:
                score += 10  # 10 points for general experience match
            
            # 3. Location match (20%)
            job_seeker_location = job_seeker_data.get('location_preference', '').lower()
            job_location = job_data.get('job_city', '').lower()
            
            if job_seeker_location and job_location:
                if job_seeker_location in job_location or job_location in job_seeker_location:
                    score += 20
                else:
                    score += 5 # Unmatched location but give base score of 5
            
            # 4. Job Title Match (20%)
            job_seeker_role = job_seeker_data.get('primary_role', '').lower()
            job_title = job_data.get('job_title', '').lower()
            
            if job_seeker_role and job_title:
                if job_seeker_role in job_title:
                    score += 20
                else:
                    # Searching for keywords in job title
                    search_terms = job_seeker_data.get('simple_search_terms', '').lower()
                    if search_terms:
                        terms = [term.strip() for term in search_terms.split(',')]
                        for term in terms:
                            if term in job_title:
                                score += 15
                                break
            
            # Make sure the score is between 0 and 100
            score = min(max(score, 0), 100)
            
            return {
                'overall_score': score,
                'matched_skills': matched_skills,
                'skill_match': len(matched_skills),
                'experience_match': 'senior' in job_seeker_experience and 'senior' in job_data.get('job_title', '').lower(),
                'location_match': job_seeker_location in job_location if job_seeker_location and job_location else False
            }
            
        except Exception as e:
            print(f"❌ Error when calculating matching score: {e}")
            return {
                'overall_score': 0,
                'matched_skills': [],
                'skill_match': 0,
                'experience_match': False,
                'location_match': False
            }

def get_all_jobs_for_matching():
    """Get all head hunter jobs for matching"""
    try:
        conn = sqlite3.connect('head_hunter_jobs.db')
        c = conn.cursor()
        c.execute("""
            SELECT id, job_title, job_description, main_responsibilities, required_skills,
                   client_company, industry, work_location, work_type, company_size,
                   employment_type, experience_level, visa_support,
                   min_salary, max_salary, currency, benefits
            FROM head_hunter_jobs
            WHERE job_valid_until >= date('now')
        """)
        jobs = c.fetchall()
        conn.close()
        return jobs
    except Exception as e:
        st.error(f"Failed to get job positions: {e}")
        return []

def get_all_job_seekers():
    """Get all job seekers information"""
    try:
        conn = sqlite3.connect('job_seeker.db')
        c = conn.cursor()
        c.execute("""
            SELECT
                id,
                education_level as education,
                work_experience as experience,
                hard_skills as skills,
                industry_preference as target_industry,
                location_preference as target_location,
                salary_expectation as expected_salary,
                university_background as current_title,
                major,
                languages,
                certificates,
                soft_skills,
                project_experience,
                benefits_expectation
            FROM job_seekers
        """)
        seekers = c.fetchall()
        conn.close()

        # Change the structure to match the expected output
        formatted_seekers = []
        for seeker in seekers:
            # Create a virtual name field (using education background + major)
            virtual_name = f"求职者#{seeker[0]} - {seeker[1]}"

            formatted_seekers.append((
                seeker[0],  # id
                virtual_name,  # name (constructed)
                seeker[3] or "",  # skills (hard_skills)
                seeker[2] or "",  # experience (work_experience)
                seeker[1] or "",  # education (education_level)
                seeker[8] or "",  # target_position (major)
                seeker[4] or "",  # target_industry (industry_preference)
                seeker[5] or "",  # target_location (location_preference)
                seeker[6] or "",  # expected_salary (salary_expectation)
                seeker[7] or ""   # current_title (university_background)
            ))

        return formatted_seekers
    except Exception as e:
        st.error(f"Failed to get job seekers: {e}")
        return []
    
def analyze_match_simple(job_data, seeker_data):
    """Simple match analysis between job and seeker"""
    match_score = 50  # Basic Score

    # Skills matching
    job_skills = str(job_data[4]).lower()
    seeker_skills = str(seeker_data[2]).lower()
    skill_match = len(set(job_skills.split()) & set(seeker_skills.split())) / max(len(job_skills.split()), 1)
    match_score += skill_match * 20

    # Experience matching
    experience_map = {"fresh graduate": 0, "1-3 years": 1, "3-5 years": 2, "5-10 years": 3, "10+ years": 4}
    #experience_map = {"应届": 0, "1-3年": 1, "3-5年": 2, "5-10年": 3, "10年以上": 4}
    job_exp = job_data[11]
    seeker_exp = seeker_data[3]

    if job_exp in experience_map and seeker_exp in experience_map:
        exp_diff = abs(experience_map[job_exp] - experience_map[seeker_exp])
        match_score -= exp_diff * 5

    # Industry matching
    job_industry = str(job_data[6]).lower()
    seeker_industry = str(seeker_data[6]).lower()
    if job_industry in seeker_industry or seeker_industry in job_industry:
        match_score += 10

    # Location matching
    job_location = str(job_data[8]).lower()
    seeker_location = str(seeker_data[7]).lower()
    if job_location in seeker_location or seeker_location in job_location:
        match_score += 5

    match_score = max(0, min(100, match_score))

    # Analyze based on score
    if match_score >= 80:
        strengths = ["High skill match", "Experience meets requirements", "Strong industry relevance"]
        #strengths = ["技能高度匹配", "经验符合要求", "行业相关性强"]
        gaps = []
        recommendation = "Highly recommend for interview"
        #recommendation = "强烈推荐面试"
    elif match_score >= 60:
        strengths = ["Core skills match", "Basic experience aligns"]
        #strengths = ["核心技能匹配", "基础经验符合"]
        gaps = ["Some skills need improvement", "Slight experience gap"]
        #gaps = ["部分技能需要提升", "经验略有差距"]
        recommendation = "Recommend further communication"
        #recommendation = "推荐进一步沟通"
    else:
        strengths = ["Has relevant background"]
        #strengths = ["有相关背景"]
        gaps = ["Low skill match", "Experience does not meet requirements"]
        #gaps = ["技能匹配度较低", "经验要求不符"]
        recommendation = "Further evaluation needed"
        #recommendation = "需要进一步评估"

    return {
        "match_score": int(match_score),
        "key_strengths": strengths,
        "potential_gaps": gaps,
        "recommendation": recommendation,
        "salary_match": "Good" if match_score > 70 else "Average",
        #"salary_match": "良好" if match_score > 70 else "一般",
        "culture_fit": "High" if match_score > 75 else "Medium"
        #"culture_fit": "高" if match_score > 75 else "中"
    }

def show_match_statistics():
    """Show match statistics"""
    st.header("📊 Match Statistics")

    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    if not jobs or not seekers:
        st.info("No statistics data available")
        return

    # Industry distribution
    st.subheader("🏭 Industry Distribution")
    industry_counts = {}
    for job in jobs:
        industry = job[6] if job[6] else "Not Specified"
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        percentage = (count / len(jobs)) * 100
        st.write(f"• **{industry}:** {count} Positions ({percentage:.1f}%)")

    # Experience Level Distribution
    st.subheader("🎯 Experience Level Distribution")
    experience_counts = {}
    for job in jobs:
        experience = job[11] if job[11] else "Not Specified"
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for exp, count in experience_counts.items():
        st.write(f"• **{exp}:** {count} Positions")

def show_instructions():
    """Display usage instructions"""
    st.header("📖 Instructions")

    st.info("""
    **Recruitment Match Instructions:**

    1. **Select Position**: Choose a position from the positions published by the headhunter module
    2. **Set Conditions**: Adjust the minimum match score and display count
    3. **Start Matching**: The system will automatically analyze the match between all job seekers and the position
    4. **View Results**: View detailed match analysis report
    5. **Take Action**: Contact candidates, schedule interviews

    **Matching Algorithm Based on:**
    • Skill Match (Hard Skills)
    • Experience Fit (Work Experience Years)
    • Industry Relevance (Industry Preferences)
    • Location Match (Work Location Preferences)
    • Comprehensive Assessment Analysis

    **Data Sources:**
    • Position Information: Positions published by Head Hunter module
    • Job Seeker Information: Information filled in Job Seeker page
    """)


def get_jobs_for_interview():
    """Get available positions for interviews"""
    try:
        conn = sqlite3.connect('head_hunter_jobs.db')
        c = conn.cursor()
        c.execute("""
            SELECT id, job_title, job_description, main_responsibilities, required_skills,
                   client_company, industry, experience_level
            FROM head_hunter_jobs
            WHERE job_valid_until >= date('now')
        """)
        jobs = c.fetchall()
        conn.close()
        return jobs
    except Exception as e:
        st.error(f"Failed to get positions: {e}")
        return []


def get_job_seeker_profile():
    """Get current job seeker information"""
    try:
        conn = sqlite3.connect('job_seeker.db')
        c = conn.cursor()
        c.execute("""
            SELECT education_level, work_experience, hard_skills, soft_skills,
                   project_experience
            FROM job_seekers
            ORDER BY id DESC
            LIMIT 1
        """)
        profile = c.fetchone()
        conn.close()
        return profile
    except Exception as e:
        st.error(f"Failed to get job seeker information: {e}")
        return None

def initialize_interview_session(job_data):
    """Initialize interview session"""
    if 'interview' not in st.session_state:
        st.session_state.interview = {
            'job_id': job_data[0],
            'job_title': job_data[1],
            'company': job_data[5],
            'current_question': 0,
            'total_questions': 2,
            'questions': [],
            'answers': [],
            'scores': [],
            'completed': False,
            'summary': None
        }

def generate_interview_question(job_data, seeker_profile, previous_qa=None):
    """Generate interview questions using Azure OpenAI"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        # Prepare position information
        job_info = f"""
Position Title: {job_data[1]}
Company: {job_data[5]}
Industry: {job_data[6]}
Experience Requirement: {job_data[7]}
Job Description: {job_data[2]}
Main Responsibilities: {job_data[3]}
Required Skills: {job_data[4]}
        """

        # Prepare job seeker information
        seeker_info = ""
        if seeker_profile:
            seeker_info = f"""
Job Seeker Background:
- Education: {seeker_profile[0]}
- Experience: {seeker_profile[1]}
- Hard Skills: {seeker_profile[2]}
- Soft Skills: {seeker_profile[3]}
- Project Experience: {seeker_profile[4]}
            """

        # Build prompt
        if previous_qa:
            prompt = f"""
As a professional interviewer, please continue the interview based on the following information:

【Position Information】
{job_info}

【Job Seeker Information】
{seeker_info}

【Previous Q&A】
Question: {previous_qa['question']}
Answer: {previous_qa['answer']}

Based on the job seeker's previous answer, please ask a relevant follow-up question. The question should:
1. Deeply explore key points from the previous answer
2. Assess the job seeker's thinking depth and professional abilities
3. Be closely related to position requirements

Please only return the question content, without additional explanations.
            """
        else:
            prompt = f"""
As a professional interviewer, please design an interview question for the following position:

【Position Information】
{job_info}

【Job Seeker Information】
{seeker_info}

Please ask a professional interview question that should:
1. Assess core abilities related to the position
2. Examine the job seeker's experience and skills
3. Have appropriate challenge level
4. Can be behavioral, technical, or situational questions

Please only return the question content, without additional explanations.
            """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional recruitment interviewer, skilled at asking targeted interview questions to assess candidates' abilities and suitability."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_tokens=500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"AI question generation failed: {str(e)}"
    
def evaluate_answer(question, answer, job_data):
    """Evaluate job seeker's answer"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        prompt = f"""
Please evaluate the following interview answer:

【Position Information】
Position: {job_data[1]}
Company: {job_data[5]}
Requirements: {job_data[4]}

【Interview Question】
{question}

【Job Seeker Answer】
{answer}

Please evaluate and provide scores (0-10 points) from the following dimensions:
1. Relevance and accuracy of the answer
2. Professional knowledge and skills demonstrated
3. Communication expression and logic
4. Match with position requirements

Please return evaluation results in the following JSON format:
{{
    "score": score,
    "feedback": "Specific feedback and suggestions",
    "strengths": ["Strength1", "Strength2"],
    "improvements": ["Improvement suggestion1", "Improvement suggestion2"]
}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional interview evaluation expert, capable of objectively assessing the quality of interview answers."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=800
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f'{{"error": "Evaluation failed: {str(e)}"}}'

def generate_final_summary(interview_data, job_data):
    """Generate final interview summary"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        # Prepare all Q&A records
        qa_history = ""
        for i, (q, a, score_data) in enumerate(zip(
            interview_data['questions'],
            interview_data['answers'],
            interview_data['scores']
        )):
            qa_history += f"""
Question {i+1}: {q}
Answer: {a}
Score: {score_data.get('score', 'N/A')}
Feedback: {score_data.get('feedback', '')}
            """


        prompt = f"""
Please generate a comprehensive summary report for the following interview:

【Position Information】
Position: {job_data[1]}
Company: {job_data[5]}
Requirements: {job_data[4]}

【Interview Q&A Records】
{qa_history}

Please provide:
1. Overall performance score (0-100 points)
2. Core strengths analysis
3. Areas needing improvement
4. Match assessment for this position
5. Specific improvement suggestions

Please return in the following JSON format:
{{
    "overall_score": overall_score,
    "summary": "Overall evaluation summary",
    "key_strengths": ["Strength1", "Strength2", "Strength3"],
    "improvement_areas": ["Improvement area1", "Improvement area2", "Improvement area3"],
    "job_fit": "High/Medium/Low",
    "recommendations": ["Recommendation1", "Recommendation2", "Recommendation3"]
}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional career advisor, capable of providing comprehensive interview performance analysis and career development suggestions."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f'{{"error": "Summary generation failed: {str(e)}"}}'

def ai_interview_page():
    """AI Interview Page"""
    st.title("🤖 AI Mock Interview")

    # Get position information
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    if not jobs:
        st.warning("❌ No available position information, please first publish positions in the headhunter module")
        return

    if not seeker_profile:
        st.warning("❌ Please first fill in your information on the Job Seeker page")
        return

    st.success("🎯 Select the position you want to interview for to start the mock interview")

    # Select position
    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("Select Interview Position", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # Display position information
    with st.expander("📋 Position Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Position:** {selected_job[1]}")
            st.write(f"**Company:** {selected_job[5]}")
            st.write(f"**Industry:** {selected_job[6]}")
        with col2:
            st.write(f"**Experience Requirement:** {selected_job[7]}")
            st.write(f"**Skill Requirements:** {selected_job[4][:100]}...")

    # Initialize interview session
    initialize_interview_session(selected_job)
    interview = st.session_state.interview

    # Start/continue interview
    if not interview['completed']:
        if interview['current_question'] == 0:
            if st.button("🚀 Start Mock Interview", type="primary", use_container_width=True):
                # Generate first question
                with st.spinner("AI is preparing interview questions..."):
                    first_question = generate_interview_question(selected_job, seeker_profile)
                    if not first_question.startswith("AI question generation failed"):
                        interview['questions'].append(first_question)
                        interview['current_question'] = 1
                        st.rerun()
                    else:
                        st.error(first_question)

        # Display current question
        if interview['current_question'] > 0 and interview['current_question'] <= interview['total_questions']:
            st.subheader(f"❓ Question {interview['current_question']}/{interview['total_questions']}")
            st.info(interview['questions'][-1])

            # Answer input
            answer = st.text_area("Your Answer:", height=150,
                                placeholder="Please describe your answer in detail...",
                                key=f"answer_{interview['current_question']}")


            if st.button("📤 Submit Answer", type="primary", use_container_width=True):
                if answer.strip():
                    with st.spinner("AI is evaluating your answer..."):
                        # Evaluate current answer
                        evaluation = evaluate_answer(
                            interview['questions'][-1],
                            answer,
                            selected_job
                        )

                        try:
                            eval_data = json.loads(evaluation)
                            if 'error' not in eval_data:
                                # Save answer and evaluation
                                interview['answers'].append(answer)
                                interview['scores'].append(eval_data)

                                # Check if all questions are completed
                                if interview['current_question'] == interview['total_questions']:
                                    # Generate final summary
                                    with st.spinner("AI is generating interview summary..."):
                                        summary = generate_final_summary(interview, selected_job)
                                        try:
                                            summary_data = json.loads(summary)
                                            interview['summary'] = summary_data
                                            interview['completed'] = True
                                        except:
                                            interview['summary'] = {"error": "Summary parsing failed"}
                                            interview['completed'] = True
                                else:
                                    # Generate next question
                                    previous_qa = {
                                        'question': interview['questions'][-1],
                                        'answer': answer
                                    }
                                    next_question = generate_interview_question(
                                        selected_job, seeker_profile, previous_qa
                                    )
                                    if not next_question.startswith("AI question generation failed"):
                                        interview['questions'].append(next_question)
                                        interview['current_question'] += 1
                                    else:
                                        st.error(next_question)

                                st.rerun()
                            else:
                                st.error(eval_data['error'])
                        except json.JSONDecodeError:
                            st.error("Evaluation result parsing failed")
                else:
                    st.warning("Please enter your answer")

            # Display progress
            progress = interview['current_question'] / interview['total_questions']
            st.progress(progress)
            st.write(f"Progress: {interview['current_question']}/{interview['total_questions']} questions")

    # Display interview results
    if interview['completed'] and interview['summary']:
        st.subheader("🎯 Interview Summary Report")

        summary = interview['summary']

        if 'error' in summary:
            st.error(summary['error'])
        else:
            # Overall score
            col1, col2, col3 = st.columns(3)
            with col1:
                score = summary.get('overall_score', 0)
                st.metric("Overall Score", f"{score}/100")
            with col2:
                st.metric("Job Fit", summary.get('job_fit', 'N/A'))
            with col3:
                st.metric("Questions Answered", f"{len(interview['answers'])}/{interview['total_questions']}")

            # Overall evaluation
            st.write("### 📊 Overall Evaluation")
            st.info(summary.get('summary', ''))

            # Core strengths
            st.write("### ✅ Core Strengths")
            strengths = summary.get('key_strengths', [])
            for strength in strengths:
                st.write(f"🎯 {strength}")

            # Improvement areas
            st.write("### 📈 Improvement Suggestions")
            improvements = summary.get('improvement_areas', [])
            for improvement in improvements:
                st.write(f"💡 {improvement}")

            # Detailed recommendations
            st.write("### 🎯 Career Development Recommendations")
            recommendations = summary.get('recommendations', [])
            for rec in recommendations:
                st.write(f"🌟 {rec}")

            # Detailed Q&A records
            with st.expander("📝 View Detailed Q&A Records"):
                for i, (question, answer, score_data) in enumerate(zip(
                    interview['questions'],
                    interview['answers'],
                    interview['scores']
                )):
                    st.write(f"#### Question {i+1}")
                    st.write(f"**Question:** {question}")
                    st.write(f"**Answer:** {answer}")
                    if isinstance(score_data, dict):
                        st.write(f"**Score:** {score_data.get('score', 'N/A')}/10")
                        st.write(f"**Feedback:** {score_data.get('feedback', '')}")
                    st.markdown("---")

            # Restart interview
            if st.button("🔄 Restart Interview", use_container_width=True):
                del st.session_state.interview
                st.rerun()