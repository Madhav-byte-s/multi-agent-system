from langchain.agents import create_agent
import langchain_core
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

from dotenv import load_dotenv

load_dotenv()
#model setup
llm = ChatMistralAI(model="ministral-8b-2512",temperature=0)


#1st agent
def build_search_agent():
   return create_agent(
      model=llm,
      tools=[web_search]
      )
#2nd agent
def build_reader_agent():
   return create_agent(
      model=llm,
      tools=[scrape_url])
#writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful rep"),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt |llm | langchain_core.output_parsers.StrOutputParser()

#critic_chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a critical reviewer. Evaluate the research report for accuracy, clarity, and depth."),
    ("human", """Critically evaluate the following research report.
    Report: {report}
    Respond in this format:
    Score: (1-10)
    Strengths: (list strengths)
    Weaknesses: (list weaknesses)
    Areas to Improve: (list areas for improvement)
    one line summary of the report's quality and reliability.
    
    """),
    
])

critic_chain = critic_prompt | llm | langchain_core.output_parsers.StrOutputParser()