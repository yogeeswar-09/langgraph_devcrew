import os
import logging
from typing import TypedDict, List

from dotenv import load_dotenv

from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, END

# ----------------------------
# Load Environment
# ----------------------------
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to your .env file.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew_agent")

# ----------------------------
# Initialize LLM
# ----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
)

# ----------------------------
# Shared Graph State
# ----------------------------
class DevCrewState(TypedDict):
    task: str
    plan: str
    code: str
    review: str
    approved: bool
    iteration: int
    max_iterations: int
    history: List[str]


# ----------------------------
# Planner Node
# ----------------------------
def planner_node(state: DevCrewState) -> DevCrewState:
    logger.info("Planner is working...")

    messages = [
        SystemMessage(
            content=(
                "You are the Planner on a software development crew. "
                "Break the user's task into a short, concrete implementation "
                "plan. Include functions/classes, edge cases, and approach. "
                "Do not write code yet."
            )
        ),
        HumanMessage(content=f"Task: {state['task']}"),
    ]

    plan = llm.invoke(messages).content
    state["plan"] = plan
    state["history"].append("Planner produced an implementation plan.")

    return state


# ----------------------------
# Coder Node
# ----------------------------
def coder_node(state: DevCrewState) -> DevCrewState:
    logger.info(
        "Coder is working - iteration %s",
        state["iteration"] + 1,
    )

    if state.get("review"):
        messages = [
            SystemMessage(
                content=(
                    "You are the Coder on a software development crew. "
                    "Revise your previous code according to the reviewer's "
                    "feedback. Return only the final code in one Python code block."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan:\n{state['plan']}\n\n"
                    f"Previous code:\n{state['code']}\n\n"
                    f"Reviewer feedback:\n{state['review']}"
                )
            ),
        ]
    else:
        messages = [
            SystemMessage(
                content=(
                    "You are the Coder on a software development crew. "
                    "Write clean, working Python code that implements the plan. "
                    "Return only the code in one Python code block."
                )
            ),
            HumanMessage(
                content=(
                    f"Task:\n{state['task']}\n\n"
                    f"Plan:\n{state['plan']}"
                )
            ),
        ]

    code = llm.invoke(messages).content
    state["code"] = code
    state["iteration"] += 1
    state["history"].append(
        f"Coder produced code (iteration {state['iteration']})."
    )

    return state


# ----------------------------
# Reviewer Node
# ----------------------------
def reviewer_node(state: DevCrewState) -> DevCrewState:
    logger.info("Reviewer is working...")

    messages = [
        SystemMessage(
            content=(
                "You are the Reviewer on a software development crew. "
                "Review the code against the task and plan for correctness, "
                "edge cases, and code quality.\n\n"
                "Respond exactly as:\n"
                "VERDICT: APPROVED or NEEDS_REVISION\n"
                "FEEDBACK: specific actionable feedback, or None if approved."
            )
        ),
        HumanMessage(
            content=(
                f"Task:\n{state['task']}\n\n"
                f"Plan:\n{state['plan']}\n\n"
                f"Code:\n{state['code']}"
            )
        ),
    ]

    result = llm.invoke(messages).content
    state["review"] = result

    # Normalize whitespace before checking the verdict.
    normalized = result.upper().replace(" ", "")
    state["approved"] = "VERDICT:APPROVED" in normalized

    state["history"].append(
        "Reviewer verdict: "
        + ("APPROVED" if state["approved"] else "NEEDS_REVISION")
        + "."
    )

    return state


# ----------------------------
# Conditional Routing
# ----------------------------
def route_after_review(state: DevCrewState) -> str:
    if state["approved"]:
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        logger.warning("Maximum iterations reached.")
        return "end"

    return "revise"


# ----------------------------
# Build LangGraph
# ----------------------------
graph_builder = StateGraph(DevCrewState)

graph_builder.add_node("planner", planner_node)
graph_builder.add_node("coder", coder_node)
graph_builder.add_node("reviewer", reviewer_node)

graph_builder.set_entry_point("planner")
graph_builder.add_edge("planner", "coder")
graph_builder.add_edge("coder", "reviewer")

graph_builder.add_conditional_edges(
    "reviewer",
    route_after_review,
    {
        "revise": "coder",
        "end": END,
    },
)

dev_crew_graph = graph_builder.compile()

# ----------------------------
# LangServe Runnable
# ----------------------------
def run_dev_crew(request):
    task = request.get("input", "")

    if not task:
        return "Please provide a software development task."

    initial_state: DevCrewState = {
        "task": task,
        "plan": "",
        "code": "",
        "review": "",
        "approved": False,
        "iteration": 0,
        "max_iterations": 3,
        "history": [],
    }

    logger.info("Starting Dev Crew for task: %s", task)

    try:
        final_state = dev_crew_graph.invoke(initial_state)

        return (
            "PLAN:\n"
            + final_state["plan"]
            + "\n\nFINAL CODE:\n"
            + final_state["code"]
            + "\n\nREVIEW:\n"
            + final_state["review"]
            + "\n\nHISTORY:\n"
            + "\n".join("- " + item for item in final_state["history"])
        )
    except Exception as exc:
        logger.exception("Dev Crew execution failed")
        return f"Dev Crew error: {exc}"


agent_runnable = RunnableLambda(run_dev_crew)

# ----------------------------
# FastAPI
# ----------------------------
app = FastAPI(
    title="LangGraph Dev Crew Agent",
    version="1.0",
    description=(
        "A multi-agent software development crew built with LangGraph: "
        "Planner, Coder, and Reviewer with a conditional revision loop."
    ),
)

add_routes(
    app,
    agent_runnable,
    path="/agent",
)

# ----------------------------
# Health Check
# ----------------------------
@app.get("/")
def root():
    return {
        "agent": "LangGraph Dev Crew Agent",
        "status": "running",
        "workflow": "Planner -> Coder -> Reviewer -> Revision",
        "endpoint": "/agent",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "langgraph_dev_crew_agent:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
    )
