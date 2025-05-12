from fastapi import APIRouter, Form
from app.services.agents import AgentService
from app.models.schemas import AgentResponse

router = APIRouter(prefix="/agents", tags=["Agents"])
service = AgentService()

@router.post("/", response_model=AgentResponse)
async def agent_task(
    task: str = Form(...),
    agents: str = Form("Researcher,Analyst,Writer,Critic,Manager"),
    model_name: str = Form("meta-llama/llama-4-scout-17b-16e-instruct"),
    process_type: str = Form("Sequential")
):
    # Convert comma-separated agents to list
    agent_list = [a.strip() for a in agents.split(",") if a.strip()]
    
    # Execute task with specified parameters
    result = service.execute_task(
        task_description=task,
        model_name=model_name,
        process_type=process_type,
        custom_agents=agent_list
    )
    
    return {
        "task": task,
        "agents_used": agent_list,
        "process_type": process_type,
        "response": result["result"],
        "status": result["status"],
        "error": result.get("error")
    }