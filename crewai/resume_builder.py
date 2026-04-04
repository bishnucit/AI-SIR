# -------------------------------
# CrewAI Job Application System (Ollama - Local)
# -------------------------------

import warnings
warnings.filterwarnings('ignore')

import os
from crewai import Agent, Task, Crew, Process

# -------------------------------
# LLM (Ollama)
# -------------------------------
llm = "ollama/mistral:7b"

# -------------------------------
# Tools
# -------------------------------
from crewai.tools import tool
from duckduckgo_search import DDGS

@tool
def search_tool(query: str) -> str:
    """Search the web and return results"""
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append(r["body"])
        return "\n".join(results)
    except Exception as e:
        return f"Search failed: {str(e)}"

from crewai_tools import FileReadTool, ScrapeWebsiteTool

# -------------------------------
# File Path (Safe)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
resume_path = os.path.join(BASE_DIR, "fake_resume.md")

read_resume = FileReadTool(file_path=resume_path)
scrape_tool = ScrapeWebsiteTool()

# -------------------------------
# Agents
# -------------------------------

researcher = Agent(
    role="Tech Job Researcher",
    goal="Extract structured job requirements",
    tools=[scrape_tool, search_tool],
    llm=llm,
    verbose=True,
    backstory="Expert in analyzing job postings and extracting key hiring signals."
)

profiler = Agent(
    role="Engineer Profile Builder",
    goal="Create a strong candidate profile",
    tools=[read_resume, search_tool],
    llm=llm,
    verbose=True,
    backstory="Specialist in building detailed technical profiles from multiple data sources."
)

resume_strategist = Agent(
    role="Resume Strategist",
    goal="Create ATS-optimized resumes",
    tools=[read_resume],
    llm=llm,
    verbose=True,
    backstory="Expert in crafting high-impact, keyword-optimized resumes."
)

interview_preparer = Agent(
    role="Interview Coach",
    goal="Prepare candidate for interviews",
    tools=[read_resume],
    llm=llm,
    verbose=True,
    backstory="Expert in generating interview questions and strong answers."
)

reviewer = Agent(
    role="Application Reviewer",
    goal="Score and critique resume",
    llm=llm,
    verbose=True,
    backstory="Expert in evaluating resumes against job requirements and identifying gaps."
)

# -------------------------------
# Tasks
# -------------------------------

research_task = Task(
    description="""
Analyze the job posting at {job_posting_url}.
If scraping fails, use search tool.
Extract:
- Required skills
- Preferred skills
- Responsibilities
- Tech stack
- Experience level
- ATS keywords
Return ONLY JSON.
""",
    expected_output="""
{
  "required_skills": [],
  "preferred_skills": [],
  "responsibilities": [],
  "tech_stack": [],
  "experience_level": "",
  "keywords": []
}
""",
    agent=researcher
)

profile_task = Task(
    description="""
Build a detailed candidate profile using:
- GitHub: {github_url}
- Personal writeup: {personal_writeup}
- Resume
Include:
- Skills
- Experience summary
- Key projects
- Leadership experience
""",
    expected_output="Structured candidate profile",
    agent=profiler
)

resume_strategy_task = Task(
    description="""
Using job requirements and candidate profile:
1. Rewrite resume tailored to the job
2. Optimize for ATS keywords
3. Highlight relevant achievements
4. Use strong action verbs
5. Keep concise
Return clean markdown resume.
""",
    expected_output="ATS-optimized resume",
    output_file="tailored_resume.md",
    context=[research_task, profile_task],
    agent=resume_strategist
)

interview_task = Task(
    description="""
Generate interview preparation:
- 10 technical questions
- 5 behavioral questions
- 5 leadership/system design questions
- Suggested answers
- Key talking points
""",
    expected_output="Interview prep material",
    output_file="interview_materials.md",
    context=[research_task, profile_task, resume_strategy_task],
    agent=interview_preparer
)

review_task = Task(
    description="""
Evaluate the resume against job requirements.
Return:
- Match score (0–100)
- Missing skills
- Strengths
- Improvements
""",
    expected_output="""
{
  "match_score": "",
  "missing_skills": [],
  "strengths": [],
  "improvements": []
}
""",
    context=[research_task, resume_strategy_task],
    agent=reviewer
)

# -------------------------------
# Crew
# -------------------------------

job_application_crew = Crew(
    agents=[
        researcher,
        profiler,
        resume_strategist,
        interview_preparer,
        reviewer
    ],
    tasks=[
        research_task,
        profile_task,
        resume_strategy_task,
        interview_task,
        review_task
    ],
    process=Process.sequential,
    memory=True,
    verbose=True
)

# -------------------------------
# Inputs
# -------------------------------

inputs = {
    'job_posting_url': 'url of job posting',
    'github_url': 'github url of user',
    'personal_writeup': """Noah is an accomplished Software Engineering Leader with 18 years of experience, specializing in managing remote and in-office teams, and expert in multiple programming languages and frameworks. He holds an MBA and a strong background in AI and data science. Noah has successfully led major tech initiatives and startups."""
}

# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":
    if not os.path.exists(resume_path):
        print("❌ ERROR: fake_resume.md not found!")
        print(f"👉 Put it here: {resume_path}")
    else:
        result = job_application_crew.kickoff(inputs=inputs)
        print("\n\nFINAL RESULT:\n")
        print(result)



"""
This script is a full AI job application assistant:

Reads your resume and GitHub.
Researches a job posting.
Builds a candidate profile.
Optimizes your resume for the job (ATS-friendly).
Generates interview prep material.
Scores and reviews your application.

Effectively, it can replace several human steps in applying for a tech job.

Sample final result  after generating tailored resume and internet materials-

FINAL RESULT:

 Here is my analysis of Noah's resume against the provided job requirements:

Evaluation Criteria:
- Match Score (0–100)
- Missing Skills
- Strengths
- Improvements

Based on the information provided in both the job posting and Noah's resume, I made the following observations:

Match Score: 85
Noah's experience and skills align well with the requirements for this role, particularly his expertise in data science, 
machine learning, programming (Python, R), SQL, and econometrics.

Noah's experience and skills align well with the requirements for this role, particularly his expertise in data science, 
machine learning, programming (Python, R), SQL, and econometrics.

metrics.


Missing Skills:
Missing Skills:
- Experience in a financial domain such as Wall Street or FinTech could strengthen Noah's application.
- Experience in a financial domain such as Wall Street or FinTech could strengthen Noah's application.
- Familiarity with quantitative modeling and econometric methods might also be beneficial, although Noah's master's degree 
would suggest that he has some understanding of these topics.
- Familiarity with quantitative modeling and econometric methods might also be beneficial, although Noah's master's degree 
would suggest that he has some understanding of these topics.

Strengths:
- Strong background in data science, machine learning, statistical computation, and programming languages is evident from Noah's professional 
experience and education.
- Proven expertise in managing teams and software projects, demonstrated through his leadership roles at Company X and Company Y.
Strengths:
- Strong background in data science, machine learning, statistical computation, and programming languages is evident from Noah's professional
experience and education.
- Proven expertise in managing teams and software projects, demonstrated through his leadership roles at Company X and Company Y.
- Strong quantitative skills as an MBA holder with a focus on econometrics, time-series analysis, and statistical computation.
- Strong background in data science, machine learning, statistical computation, and programming languages is evident from Noah's professional 
experience and education.
- Proven expertise in managing teams and software projects, demonstrated through his leadership roles at Company X and Company Y.
- Strong quantitative skills as an MBA holder with a focus on econometrics, time-series analysis, and statistical computation.


Improvements:
Improvements:
Noah may wish to emphasize in his resume specific examples of the impact and results he has achieved through his work, 
particularly in the financial sector or related to predictiNoah may wish to emphasize in his resume specific examples of the impact and 
results he has achieved through his work, particularly in the financial sector or related to predictive modeling, econometric methods, 
and visualization tools. Additionally, highlighting any participation in open-source projects could further bolster his application.


Tailored_resume.md

 Here's a tailored resume for Noah:

---

Noah Noah
==============
Software Engineering Leader | AI & Data Science Specialist | 18 years experience

Contact Information:
- Phone: +1 (555) 555-5555
- Email: [noah@email.com](mailto:noah@email.com)
- LinkedIn: linkedin.com/in/noahmoura
- GitHub: github.com/joaomdmoura

Summary:
Expert Software Engineering Leader with 18 years of experience managing remote and in-office teams. Specializing in AI, data science, and multiple programming languages, including Python, C++, and JavaScript. Holds an MBA. Focusing on creating ATS-optimized, customized resumes to help applicants stand out and land interviews.

Professional Experience:
Machine Learning Engineer | [Company X] | [Location] | [Timeframe]
- Conducted cutting-edge research in data analysis and machine learning applied to finance (Keyword Match)
- Developed advanced predictive models, econometric methods, statistical inference techniques, algorithms, and visualization tools (Keyword Match)
- Collaborated with the investment team to identify new market opportunities and inform strategic investment decisions (Keyword Match)
- Executed quantitative projects using both internal data and external data sources (Keyword Match)
- Built knowledge base of best practices for project management, ensuring continuous improvement of data infrastructure and software architecture (Keyword Match)

Senior Data Scientist | [Company Y] | [Location] | [Timeframe]
- Led initiatives by executing deep learning projects to create and improve predictive models, algorithms, and visualization tools
- Managed teams to execute ETL processes for data integration and web scraping efforts in support of company goals
- Developed custom software solutions leveraging Apache Airflow for improved efficiency of tasks and workflows (Keyword Match)

Education:
Master of Business Administration | [University] | [Location] | [Timeframe]
- Acquired expertise in econometrics, time-series analysis, and statistical computation using R, SAS, and Stata
Bachelor of Science in Computer Engineering | [University] | [Location] | [Timeframe]
- Focused curriculum in programming, data structures, and algorithms

 Skills:
- Python (Strong)
- Machine Learning
- Data Science
- SQL, Database Management & Querying
- R, SAS, Stata
- Web Scraping, ETL Processes, Distributed Computing (Apache Airflow)
- Causal Inference, Econometrics, Time-Series Analysis
- JavaScript, C++, Java
- Financial Models, Financial Analysis, Financial Forecasting
- Leadership & Project Management

---



interview_maaterials.md


 Here is the interview preparation for Noah, tailored around the job posting provided:

----
**Technical Questions:**
1. Explain your experience with data analytics and machine learning in the context of finance. (Expected answer: Share past projects or experiences where you've applied your skills to financial analysis)
2. What are some common econometric methods you use for statistical inference, and describe a time when their application significantly improved a project result. (Expected answer: Mention specific methods such as regression analysis, time-series forecasting, and provide examples of where they have been successfully applied)
3. How do you effectively prepare, clean, and preprocess large datasets for statistical computation? (Expected answer: Describe your approach to handling missing data, outliers, or noisy values in a dataset)
4. What is your familiarity with SQL and Apache Airflow, and how have they been utilized on a project recently? Can you provide examples of the benefits obtained by their use? (Expected answer: Explain projects where these technologies were implemented, emphasizing increased efficiency or streamlined processes)
5. Describe a situation where data visualization played an important role in your work, and what tools/libraries did you utilize to generate meaningful insights? (Expected answer: Share examples of Seaborn, Matplotlib, or other libraries used for creating insightful charts and graphs that informed decision-making processes)
6. Have you had experience contributing to open source projects or communities related to machine learning, data science, or quantitative finance? If so, please describe one project with its significance and impact (Optional but strong if provided)
7. How do you stay current on the latest research, methodologies, and best practices in your field? Please discuss a few resources that you follow or find valuable for staying informed.
8. Explain how you can build, validate, and interpret machine learning models to make accurate predictions about financial markets. What are some model evaluation techniques you use to ensure their efficacy (Optional but good for an advanced question)
9. What steps do you take to ensure the security and integrity of data when working with external sources or APIs? (Expected answer: Discuss how access keys, authentication tokens, and other measures are employed for secure data integration)
10. How would you approach a collaborative project in which your machine learning model needs to work alongside human decision-makers? Describe strategies for creating an intuitive interface that facilitates their interaction with the model (Optional, if relevant to the company's work)

----
**Behavioral Questions:**
1. Can you provide an example of a time when you overcame a significant challenge or obstacle while working on data analysis and machine learning projects? (Expected answer: Describe a difficult problem that was resolved with determination and creativity, accompanied by specific actions taken to overcome the hurdle)
2. How do you adapt your communication style to effectively collaborate with non-technical colleagues in explaining complex concepts or results from your research? Provide an example of the process in action. (Expected answer: Share a case where you successfully articulated key findings and their implications to non-technical team members)
3. Describe a team dynamic that you've found particularly productive, and explain how you contributed to its success. (Expected answer: Include specific actions taken to foster collaboration, clear communication, or problem-solving practices within the team setting)
4. Can you offer an instance where your ability to manage deadlines effectively resulted in a positive outcome for the project's success? (Expected answer: Describe how prioritization skills and time management contributed to delivering high-quality results within schedule constraints)
5. How do you handle constructive feedback during your work, and are there any strategies you use to improve yourself based on this input? Please give an example where feedback positively impacted a past project (Expected answer: Share experiences of embracing constructive criticism and learning from it, demonstrating self-improvement)

----
**Leadership/System Design Questions:**
1. Describe your experience leading teams with a focus on technology initiatives or startups? What strategies did you employ to foster innovation and efficiency in the given environment? (Expected answer: Share examples of problem-solving techniques, communication practices, and project management methodologies employed within the team setting)
2. Outline a time when your expertise helped drive strategic investment decisions in finance by providing data-driven insights and actionable recommendations. Discuss specific steps taken to inform decision-makers. (Expected answer: Describe how quantitative analysis and modeling techniques guided investment strategies, resulting in positive outcomes)
3. If you were tasked with improving the efficiency of workflows and tasks within our organization using distributed computing frameworks like Apache Airflow, what is your approach for evaluating our current state, identifying pain points, and proposing potential solutions? (Expected answer: Share steps such as analyzing existing processes, identifying bottlenecks, proposing improvements through workflow reconfiguration or automation)
4. Describe a system design project in which you made significant contributions to the architecture, development process, or technical decisions that substantially enhanced the end product. What role did you play and what were the project's overall achievements? (Expected answer: Discuss your involvement in a complex system design problem, explaining how your contributions helped achieve quality results and a robust solution)
5. How do you balance creative thinking to drive innovation with maintaining the highest standards of data privacy and security practices within your projects or collaborations? Please share examples that demonstrate this delicate balance. (Expected answer: Discuss specific strategies for striking a balance between fostering creativity and ensuring proper data protection measures, including guidelines such as anonymization techniques for sensitive information)

----


"""
