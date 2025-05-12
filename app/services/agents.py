import os
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
import google.generativeai as genai

# Load environment variables
load_dotenv()

class AgentService:
    def __init__(self):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")

    def _create_groq_llm(self, model_name: str):
        """Create Groq LLM instance for CrewAI"""
        return ChatGroq(
            groq_api_key=self.groq_api_key,
            model_name=model_name,
            temperature=0.7,
            max_tokens=1024
        )

    def _fallback_to_gemini(self, task_description: str):
        """Fallback to Gemini model when primary model fails"""
        try:
            client = genai.Client(api_key=self.google_api_key)
            prompt = f"""
            Complete this task as a team of experts:
            1. Research: Gather all relevant information
            2. Analyze: Identify key insights and patterns
            3. Synthesize: Create final comprehensive output
            
            Task: {task_description}
            
            Format response with:
            - Research Findings
            - Analysis & Insights
            - Final Deliverable
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro-exp-03-25",
                contents=[prompt]
            )
            return response.text
        except Exception as e:
            return f"Fallback error: {str(e)}"

    def execute_task(
        self,
        task_description: str,
        model_name: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        process_type: str = "Sequential",
        custom_agents: List[str] = None
    ) -> Dict:
        """Execute agentic task with CrewAI framework"""
        try:
            # Set default agents if none provided
            if not custom_agents:
                custom_agents = ["Researcher", "Analyst", "Writer"]

            # Create LLM instance
            llm = self._create_groq_llm(model_name)
            
            agents = []
            tasks = []

            # Agent creation logic
            if "Researcher" in custom_agents:
                researcher = Agent(
                    role="Senior Researcher",
                    goal="Gather comprehensive information for the task",
                    backstory="Expert in thorough research and data collection",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(researcher)
                tasks.append(Task(
                    description=f"Research: {task_description}",
                    agent=researcher,
                    expected_output="Detailed research findings"
                ))

            if "Analyst" in custom_agents:
                analyst = Agent(
                    role="Chief Analyst",
                    goal="Analyze information and extract insights",
                    backstory="Skilled in pattern recognition and critical analysis",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(analyst)
                tasks.append(Task(
                    description=f"Analyze research for: {task_description}",
                    agent=analyst,
                    expected_output="Key insights and analysis report",
                    context=tasks.copy()
                ))

            if "Writer" in custom_agents:
                writer = Agent(
                    role="Lead Writer",
                    goal="Produce high-quality written content",
                    backstory="Expert technical writer and communicator",
                    verbose=True,
                    allow_delegation=False,
                    llm=llm
                )
                agents.append(writer)
                tasks.append(Task(
                    description=f"Create final output for: {task_description}",
                    agent=writer,
                    expected_output="Polished final deliverable",
                    context=tasks.copy()
                ))

            if "Manager" in custom_agents and process_type == "Hierarchical":
                manager = Agent(
                    role="Project Manager",
                    goal="Coordinate team activities",
                    backstory="Experienced project coordinator",
                    verbose=True,
                    allow_delegation=True,
                    llm=llm
                )
                agents.insert(0, manager)
                tasks = [Task(
                    description=f"Manage execution of: {task_description}",
                    agent=manager,
                    expected_output="Completed project deliverables"
                )]

            # Configure process
            process = Process.SEQUENTIAL if process_type == "Sequential" else Process.HIERARCHICAL
            
            # Execute crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                verbose=True,
                process=process
            )
            
            result = crew.kickoff()

            # Fallback if empty result
            if not result.strip():
                result = self._fallback_to_gemini(task_description)
                return {
                    "result": f"[FALLBACK] {result}",
                    "status": "fallback"
                }

            return {
                "result": result,
                "agents": custom_agents,
                "process_type": process_type,
                "status": "success"
            }

        except Exception as e:
            # Attempt Gemini fallback
            try:
                fallback_result = self._fallback_to_gemini(task_description)
                return {
                    "result": f"[FALLBACK] {fallback_result}",
                    "error": str(e),
                    "status": "fallback"
                }
            except Exception as gemini_error:
                return {
                    "result": f"Both models failed: {str(e)} | {str(gemini_error)}",
                    "status": "error"
                }