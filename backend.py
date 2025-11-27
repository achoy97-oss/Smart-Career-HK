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
以下是候选人的完整简历内容，请从中提取结构化信息（如果缺失请留空）：
cv_text: '''{cv_text}'''

请输出 JSON，字段包括：
- education_level（博士/硕士/本科/大专/高中）
- major
- graduation_status（应届生/往届生/在读）
- university_background（985院校/211院校/海外院校/普通本科/其他）
- languages
- certificates
- hard_skills
- soft_skills
- work_experience（应届/1-3年/3-5年/5-10年/10年以上）
- project_experience
- location_preference
- industry_preference
- salary_expectation
- benefits_expectation

请直接返回 JSON，不要解释。
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
        """从JSearch API获取真实职位数据"""
        try:
            # JSearch API配置
            API_KEY = "your_jsearch_api_key_here"  # 你需要从 https://jsearch.app/ 获取API密钥
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
                    print(f"✅ 第 {page} 页获取到 {len(jobs)} 个职位")
                else:
                    print(f"❌ API请求失败: {response.status_code}")
                    break
                    
            print(f"🎯 总共获取到 {len(all_jobs)} 个职位")
            return all_jobs
            
        except Exception as e:
            print(f"❌ 获取职位数据失败: {e}")
            # 返回模拟数据作为备选
            return self.get_mock_jobs(search_query, location)

    def get_mock_jobs(self, search_query, location):
        """返回模拟职位数据（当API不可用时使用）"""
        print("🔄 使用模拟数据...")
        
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
        """计算职位匹配分数"""
        try:
            score = 0
            max_score = 100
            matched_skills = []
            
            # 1. 技能匹配 (40分)
            job_seeker_skills = job_seeker_data.get('hard_skills', '').lower()
            job_description = job_data.get('job_description', '').lower()
            
            if job_seeker_skills:
                skills_list = [skill.strip().lower() for skill in job_seeker_skills.split(',')]
                for skill in skills_list:
                    if skill and skill in job_description:
                        score += 5  # 每个匹配的技能加5分
                        matched_skills.append(skill)
                        if score >= 40:  # 技能分上限40分
                            score = 40
                            break
            
            # 2. 经验匹配 (20分)
            job_seeker_experience = job_seeker_data.get('work_experience', '').lower()
            if 'senior' in job_data.get('job_title', '').lower() and 'senior' in job_seeker_experience.lower():
                score += 20
            elif 'junior' in job_data.get('job_title', '').lower() and 'junior' in job_seeker_experience.lower():
                score += 20
            elif 'entry' in job_data.get('job_title', '').lower() and 'fresh' in job_seeker_experience.lower():
                score += 20
            else:
                score += 10  # 基础经验分
            
            # 3. 地点匹配 (20分)
            job_seeker_location = job_seeker_data.get('location_preference', '').lower()
            job_location = job_data.get('job_city', '').lower()
            
            if job_seeker_location and job_location:
                if job_seeker_location in job_location or job_location in job_seeker_location:
                    score += 20
                else:
                    score += 5  # 地点不匹配但给基础分
            
            # 4. 职位名称匹配 (20分)
            job_seeker_role = job_seeker_data.get('primary_role', '').lower()
            job_title = job_data.get('job_title', '').lower()
            
            if job_seeker_role and job_title:
                if job_seeker_role in job_title:
                    score += 20
                else:
                    # 检查搜索关键词匹配
                    search_terms = job_seeker_data.get('simple_search_terms', '').lower()
                    if search_terms:
                        terms = [term.strip() for term in search_terms.split(',')]
                        for term in terms:
                            if term in job_title:
                                score += 15
                                break
            
            # 确保分数在0-100之间
            score = min(max(score, 0), 100)
            
            return {
                'overall_score': score,
                'matched_skills': matched_skills,
                'skill_match': len(matched_skills),
                'experience_match': 'senior' in job_seeker_experience and 'senior' in job_data.get('job_title', '').lower(),
                'location_match': job_seeker_location in job_location if job_seeker_location and job_location else False
            }
            
        except Exception as e:
            print(f"❌ 计算匹配分数时出错: {e}")
            return {
                'overall_score': 0,
                'matched_skills': [],
                'skill_match': 0,
                'experience_match': False,
                'location_match': False
            }

def get_all_jobs_for_matching():
    """获取所有猎头职位用于匹配"""
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
        st.error(f"获取职位失败: {e}")
        return []

def get_all_job_seekers():
    """获取所有求职者信息"""
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

        # 转换数据格式以匹配原有结构
        formatted_seekers = []
        for seeker in seekers:
            # 创建虚拟name字段（使用教育背景+专业）
            virtual_name = f"求职者#{seeker[0]} - {seeker[1]}"

            formatted_seekers.append((
                seeker[0],  # id
                virtual_name,  # name (虚拟)
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
        st.error(f"获取求职者失败: {e}")
        return []
    
def analyze_match_simple(job_data, seeker_data):
    """简化版匹配分析"""
    match_score = 50  # 基础分数

    # 技能匹配
    job_skills = str(job_data[4]).lower()
    seeker_skills = str(seeker_data[2]).lower()
    skill_match = len(set(job_skills.split()) & set(seeker_skills.split())) / max(len(job_skills.split()), 1)
    match_score += skill_match * 20

    # 经验匹配
    experience_map = {"应届": 0, "1-3年": 1, "3-5年": 2, "5-10年": 3, "10年以上": 4}
    job_exp = job_data[11]
    seeker_exp = seeker_data[3]

    if job_exp in experience_map and seeker_exp in experience_map:
        exp_diff = abs(experience_map[job_exp] - experience_map[seeker_exp])
        match_score -= exp_diff * 5

    # 行业匹配
    job_industry = str(job_data[6]).lower()
    seeker_industry = str(seeker_data[6]).lower()
    if job_industry in seeker_industry or seeker_industry in job_industry:
        match_score += 10

    # 地点匹配
    job_location = str(job_data[8]).lower()
    seeker_location = str(seeker_data[7]).lower()
    if job_location in seeker_location or seeker_location in job_location:
        match_score += 5

    match_score = max(0, min(100, match_score))

    # 根据分数生成分析
    if match_score >= 80:
        strengths = ["技能高度匹配", "经验符合要求", "行业相关性强"]
        gaps = []
        recommendation = "强烈推荐面试"
    elif match_score >= 60:
        strengths = ["核心技能匹配", "基础经验符合"]
        gaps = ["部分技能需要提升", "经验略有差距"]
        recommendation = "推荐进一步沟通"
    else:
        strengths = ["有相关背景"]
        gaps = ["技能匹配度较低", "经验要求不符"]
        recommendation = "需要进一步评估"

    return {
        "match_score": int(match_score),
        "key_strengths": strengths,
        "potential_gaps": gaps,
        "recommendation": recommendation,
        "salary_match": "良好" if match_score > 70 else "一般",
        "culture_fit": "高" if match_score > 75 else "中"
    }

def show_match_statistics():
    """显示匹配统计"""
    st.header("📊 匹配统计")

    jobs = get_all_jobs_for_matching()
    seekers = get_all_job_seekers()

    if not jobs or not seekers:
        st.info("暂无统计数据")
        return

    # 行业分布
    st.subheader("🏭 职位行业分布")
    industry_counts = {}
    for job in jobs:
        industry = job[6] if job[6] else "未分类"
        industry_counts[industry] = industry_counts.get(industry, 0) + 1

    for industry, count in industry_counts.items():
        percentage = (count / len(jobs)) * 100
        st.write(f"• **{industry}:** {count} 个职位 ({percentage:.1f}%)")

    # 经验要求分布
    st.subheader("🎯 经验要求分布")
    experience_counts = {}
    for job in jobs:
        experience = job[11] if job[11] else "未指定"
        experience_counts[experience] = experience_counts.get(experience, 0) + 1

    for exp, count in experience_counts.items():
        st.write(f"• **{exp}:** {count} 个职位")

def show_instructions():
    """显示使用说明"""
    st.header("📖 使用说明")

    st.info("""
    **Recruitment Match 使用指南:**

    1. **选择职位**: 从猎头模块发布的职位中选择一个进行匹配
    2. **设置条件**: 调整最低匹配分数和显示数量
    3. **开始匹配**: 系统会自动分析所有求职者与职位的匹配度
    4. **查看结果**: 查看详细的匹配分析报告
    5. **采取行动**: 联系候选人、安排面试

    **匹配算法基于:**
    • 技能匹配度 (硬技能)
    • 经验符合度 (工作经验年限)
    • 行业相关性 (行业偏好)
    • 地点匹配度 (工作地点偏好)
    • 综合评估分析

    **数据来源:**
    • 职位信息: Head Hunter 模块发布的职位
    • 求职者信息: Job Seeker 页面填写的信息
    """)

def get_jobs_for_interview():
    """获取可用于面试的职位"""
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
        st.error(f"获取职位失败: {e}")
        return []

def get_job_seeker_profile():
    """获取当前求职者信息"""
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
        st.error(f"获取求职者信息失败: {e}")
        return None

def initialize_interview_session(job_data):
    """初始化面试会话"""
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
    """使用Azure OpenAI生成面试问题"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        # 准备职位信息
        job_info = f"""
职位标题: {job_data[1]}
公司: {job_data[5]}
行业: {job_data[6]}
经验要求: {job_data[7]}
职位描述: {job_data[2]}
主要职责: {job_data[3]}
必备技能: {job_data[4]}
        """

        # 准备求职者信息
        seeker_info = ""
        if seeker_profile:
            seeker_info = f"""
求职者背景:
- 教育: {seeker_profile[0]}
- 经验: {seeker_profile[1]}
- 硬技能: {seeker_profile[2]}
- 软技能: {seeker_profile[3]}
- 项目经验: {seeker_profile[4]}
            """

        # 构建提示词
        if previous_qa:
            prompt = f"""
作为专业的面试官，请基于以下信息继续面试：

【职位信息】
{job_info}

【求职者信息】
{seeker_info}

【之前的问答】
问题: {previous_qa['question']}
回答: {previous_qa['answer']}

请基于求职者的上一个回答，提出一个相关的跟进问题。问题应该：
1. 深入探讨上一个回答中的关键点
2. 评估求职者的思考深度和专业能力
3. 与职位要求紧密相关

请只返回问题内容，不要添加其他说明。
            """
        else:
            prompt = f"""
作为专业的面试官，请为以下职位设计一个面试问题：

【职位信息】
{job_info}

【求职者信息】
{seeker_info}

请提出一个专业的面试问题，问题应该：
1. 评估与职位相关的核心能力
2. 考察求职者的经验和技能
3. 具有适当的挑战性
4. 可以是行为面试问题、技术问题或情景问题

请只返回问题内容，不要添加其他说明。
            """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的招聘面试官，擅长提出有针对性的面试问题来评估候选人的能力和适应性。"
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
        return f"AI问题生成失败: {str(e)}"

def evaluate_answer(question, answer, job_data):
    """评估求职者的回答"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        prompt = f"""
请评估以下面试回答：

【职位信息】
职位: {job_data[1]}
公司: {job_data[5]}
要求: {job_data[4]}

【面试问题】
{question}

【求职者回答】
{answer}

请从以下维度评估并给出分数（0-10分）：
1. 回答的相关性和准确性
2. 展示的专业知识和技能
3. 沟通表达和逻辑性
4. 与职位要求的匹配度

请用以下JSON格式返回评估结果：
{{
    "score": 分数,
    "feedback": "具体反馈和建议",
    "strengths": ["优势1", "优势2"],
    "improvements": ["改进建议1", "改进建议2"]
}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的面试评估专家，能够客观评估面试回答的质量。"
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
        return f'{{"error": "评估失败: {str(e)}"}}'

def generate_final_summary(interview_data, job_data):
    """生成最终面试总结"""
    try:
        client = AzureOpenAI(
            azure_endpoint="https://hkust.azure-api.net",
            api_version="2024-10-21",
            api_key="7b567f8243bc4985a4e1f870092a3e60"
        )

        # 准备所有问答记录
        qa_history = ""
        for i, (q, a, score_data) in enumerate(zip(
            interview_data['questions'],
            interview_data['answers'],
            interview_data['scores']
        )):
            qa_history += f"""
问题 {i+1}: {q}
回答: {a}
评分: {score_data.get('score', 'N/A')}
反馈: {score_data.get('feedback', '')}
            """

        prompt = f"""
请为以下面试生成全面的总结报告：

【职位信息】
职位: {job_data[1]}
公司: {job_data[5]}
要求: {job_data[4]}

【面试问答记录】
{qa_history}

请提供：
1. 总体表现评分（0-100分）
2. 核心优势分析
3. 需要改进的领域
4. 针对该职位的匹配度评估
5. 具体的提升建议

请用以下JSON格式返回：
{{
    "overall_score": 总体分数,
    "summary": "总体评价总结",
    "key_strengths": ["优势1", "优势2", "优势3"],
    "improvement_areas": ["改进领域1", "改进领域2", "改进领域3"],
    "job_fit": "高/中/低",
    "recommendations": ["建议1", "建议2", "建议3"]
}}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的职业顾问，能够提供全面的面试表现分析和职业发展建议。"
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
        return f'{{"error": "总结生成失败: {str(e)}"}}'

def ai_interview_page():
    """AI面试页面"""
    st.title("🤖 AI模拟面试")

    # 获取职位信息
    jobs = get_jobs_for_interview()
    seeker_profile = get_job_seeker_profile()

    if not jobs:
        st.warning("❌ 没有可用的职位信息，请先在猎头模块发布职位")
        return

    if not seeker_profile:
        st.warning("❌ 请先在Job Seeker页面填写您的信息")
        return

    st.success("🎯 选择您想要面试的职位开始模拟面试")

    # 选择职位
    job_options = {f"#{job[0]} {job[1]} - {job[5]}": job for job in jobs}
    selected_job_key = st.selectbox("选择面试职位", list(job_options.keys()))
    selected_job = job_options[selected_job_key]

    # 显示职位信息
    with st.expander("📋 职位信息", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**职位:** {selected_job[1]}")
            st.write(f"**公司:** {selected_job[5]}")
            st.write(f"**行业:** {selected_job[6]}")
        with col2:
            st.write(f"**经验要求:** {selected_job[7]}")
            st.write(f"**技能要求:** {selected_job[4][:100]}...")

    # 初始化面试会话
    initialize_interview_session(selected_job)
    interview = st.session_state.interview

    # 开始/继续面试
    if not interview['completed']:
        if interview['current_question'] == 0:
            if st.button("🚀 开始模拟面试", type="primary", use_container_width=True):
                # 生成第一个问题
                with st.spinner("AI正在准备面试问题..."):
                    first_question = generate_interview_question(selected_job, seeker_profile)
                    if not first_question.startswith("AI问题生成失败"):
                        interview['questions'].append(first_question)
                        interview['current_question'] = 1
                        st.rerun()
                    else:
                        st.error(first_question)

        # 显示当前问题
        if interview['current_question'] > 0 and interview['current_question'] <= interview['total_questions']:
            st.subheader(f"❓ 问题 {interview['current_question']}/{interview['total_questions']}")
            st.info(interview['questions'][-1])

            # 回答输入
            answer = st.text_area("您的回答:", height=150,
                                placeholder="请详细描述您的回答...",
                                key=f"answer_{interview['current_question']}")

            if st.button("📤 提交回答", type="primary", use_container_width=True):
                if answer.strip():
                    with st.spinner("AI正在评估您的回答..."):
                        # 评估当前回答
                        evaluation = evaluate_answer(
                            interview['questions'][-1],
                            answer,
                            selected_job
                        )

                        try:
                            eval_data = json.loads(evaluation)
                            if 'error' not in eval_data:
                                # 保存回答和评估
                                interview['answers'].append(answer)
                                interview['scores'].append(eval_data)

                                # 检查是否完成所有问题
                                if interview['current_question'] == interview['total_questions']:
                                    # 生成最终总结
                                    with st.spinner("AI正在生成面试总结..."):
                                        summary = generate_final_summary(interview, selected_job)
                                        try:
                                            summary_data = json.loads(summary)
                                            interview['summary'] = summary_data
                                            interview['completed'] = True
                                        except:
                                            interview['summary'] = {"error": "总结解析失败"}
                                            interview['completed'] = True
                                else:
                                    # 生成下一个问题
                                    previous_qa = {
                                        'question': interview['questions'][-1],
                                        'answer': answer
                                    }
                                    next_question = generate_interview_question(
                                        selected_job, seeker_profile, previous_qa
                                    )
                                    if not next_question.startswith("AI问题生成失败"):
                                        interview['questions'].append(next_question)
                                        interview['current_question'] += 1
                                    else:
                                        st.error(next_question)

                                st.rerun()
                            else:
                                st.error(eval_data['error'])
                        except json.JSONDecodeError:
                            st.error("评估结果解析失败")
                else:
                    st.warning("请输入您的回答")

            # 显示进度
            progress = interview['current_question'] / interview['total_questions']
            st.progress(progress)
            st.write(f"进度: {interview['current_question']}/{interview['total_questions']} 题")

    # 显示面试结果
    if interview['completed'] and interview['summary']:
        st.subheader("🎯 面试总结报告")

        summary = interview['summary']

        if 'error' in summary:
            st.error(summary['error'])
        else:
            # 总体评分
            col1, col2, col3 = st.columns(3)
            with col1:
                score = summary.get('overall_score', 0)
                st.metric("总体评分", f"{score}/100")
            with col2:
                st.metric("职位匹配度", summary.get('job_fit', 'N/A'))
            with col3:
                st.metric("回答问题", f"{len(interview['answers'])}/{interview['total_questions']}")

            # 总体评价
            st.write("### 📊 总体评价")
            st.info(summary.get('summary', ''))

            # 核心优势
            st.write("### ✅ 核心优势")
            strengths = summary.get('key_strengths', [])
            for strength in strengths:
                st.write(f"🎯 {strength}")

            # 改进领域
            st.write("### 📈 改进建议")
            improvements = summary.get('improvement_areas', [])
            for improvement in improvements:
                st.write(f"💡 {improvement}")

            # 详细建议
            st.write("### 🎯 职业发展建议")
            recommendations = summary.get('recommendations', [])
            for rec in recommendations:
                st.write(f"🌟 {rec}")

            # 详细问答记录
            with st.expander("📝 查看详细问答记录"):
                for i, (question, answer, score_data) in enumerate(zip(
                    interview['questions'],
                    interview['answers'],
                    interview['scores']
                )):
                    st.write(f"#### 问题 {i+1}")
                    st.write(f"**问题:** {question}")
                    st.write(f"**回答:** {answer}")
                    if isinstance(score_data, dict):
                        st.write(f"**评分:** {score_data.get('score', 'N/A')}/10")
                        st.write(f"**反馈:** {score_data.get('feedback', '')}")
                    st.markdown("---")

            # 重新开始面试
            if st.button("🔄 重新开始面试", use_container_width=True):
                del st.session_state.interview
                st.rerun()
