import os
import logging
from typing import TypedDict

from dotenv import load_dotenv
from fastapi import FastAPI
from langserve import add_routes
import uvicorn

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda

from langgraph.graph import StateGraph, END


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dev_crew")

GOOGLE_API_KEY = (
    os.getenv("GOOGLE_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)

if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY or GEMINI_API_KEY is missing."
    )


# ============================================================
# LLM
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    max_retries=0,
    timeout=60,
)


# ============================================================
# SHARED LANGGRAPH STATE
# ============================================================

class DevCrewState(TypedDict, total=False):

    user_request: str

    requirements: str

    architecture: str

    implementation: str

    testing: str

    review: str

    final_report: str


# ============================================================
# HELPER
# ============================================================

def ask_agent(role: str, task: str) -> str:

    prompt = f"""
You are the {role} in an AI software development team called Dev Crew.

Your responsibility:
{task}

Work only on your assigned responsibility.

Do not reveal chain-of-thought.

Return a concise professional result that another team member
can use as input.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    parts.append(
                        str(item.get("text", ""))
                    )

                else:

                    parts.append(str(item))

            content = "\n".join(parts)

        return str(content).strip()

    except Exception as error:

        logger.exception("%s failed", role)

        return (
            f"{role} could not complete its task.\n"
            f"Error: {error}"
        )


# ============================================================
# AGENT 1
# REQUIREMENTS AGENT
# ============================================================

def requirements_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 1 -> Requirements Agent")

    request = state["user_request"]

    result = ask_agent(
        "Requirements Agent",
        f"""
Analyze this software development request:

{request}

Identify:

1. Functional requirements
2. Non-functional requirements
3. User roles
4. Inputs and outputs
5. Important constraints
6. Possible edge cases
"""
    )

    return {
        "requirements": result
    }


# ============================================================
# AGENT 2
# ARCHITECT AGENT
# ============================================================

def architect_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 2 -> Architect Agent")

    result = ask_agent(
        "Software Architect",
        f"""
Design the architecture for this request:

USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

Define:

1. Recommended technology stack
2. Application architecture
3. Main modules
4. API structure
5. Database design where required
6. Data flow
7. Security considerations
8. Deployment approach
"""
    )

    return {
        "architecture": result
    }


# ============================================================
# AGENT 3
# DEVELOPER AGENT
# ============================================================

def developer_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 3 -> Developer Agent")

    result = ask_agent(
        "Senior Developer",
        f"""
Develop the solution based on:

USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

ARCHITECTURE:
{state["architecture"]}

Provide:

1. Project structure
2. Important implementation details
3. Relevant code examples
4. API implementation where appropriate
5. Database implementation where appropriate
6. Configuration requirements
7. Error handling
"""
    )

    return {
        "implementation": result
    }


# ============================================================
# AGENT 4
# TESTING AGENT
# ============================================================

def testing_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 4 -> Testing Agent")

    result = ask_agent(
        "QA and Testing Engineer",
        f"""
Create a testing strategy for this project.

USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

ARCHITECTURE:
{state["architecture"]}

IMPLEMENTATION:
{state["implementation"]}

Identify:

1. Unit tests
2. API tests
3. Integration tests
4. Validation tests
5. Security tests
6. Edge cases
7. Expected results
"""
    )

    return {
        "testing": result
    }


# ============================================================
# AGENT 5
# REVIEWER AGENT
# ============================================================

def reviewer_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 5 -> Reviewer Agent")

    result = ask_agent(
        "Senior Code Reviewer",
        f"""
Perform a technical review.

USER REQUEST:
{state["user_request"]}

REQUIREMENTS:
{state["requirements"]}

ARCHITECTURE:
{state["architecture"]}

IMPLEMENTATION:
{state["implementation"]}

TESTING:
{state["testing"]}

Review:

1. Correctness
2. Requirement coverage
3. Security
4. Maintainability
5. Scalability
6. Error handling
7. Testing coverage
8. Deployment readiness

Give specific improvements.
"""
    )

    return {
        "review": result
    }


# ============================================================
# AGENT 6
# LEAD / FINALIZER AGENT
# ============================================================

def lead_agent(
    state: DevCrewState
) -> DevCrewState:

    logger.info("AGENT 6 -> Lead Agent")

    prompt = f"""
You are the Lead Engineer of Dev Crew.

Create the final professional development report.

USER REQUEST:
{state["user_request"]}

REQUIREMENTS AGENT:
{state["requirements"]}

ARCHITECT AGENT:
{state["architecture"]}

DEVELOPER AGENT:
{state["implementation"]}

TESTING AGENT:
{state["testing"]}

REVIEWER AGENT:
{state["review"]}

Create one coherent final report.

Use exactly this structure:

# Dev Crew Final Report

## 1. Project Understanding

Explain what needs to be built.

## 2. Requirements

Summarize the important functional and non-functional requirements.

## 3. Architecture

Explain the recommended architecture and technology stack.

## 4. Implementation

Explain the implementation approach and provide important code
or project structure where useful.

## 5. Testing Strategy

Explain how the application should be tested.

## 6. Technical Review

Summarize risks, issues and recommended improvements.

## 7. Deployment

Explain how the solution can be deployed.

## 8. Final Recommendation

Give a concise professional conclusion.

IMPORTANT:

- Do not reveal chain-of-thought.
- Do not claim code was executed or tested.
- Do not invent test results.
- Clearly distinguish recommendations from completed work.
- Make the report understandable to a student developer.
"""

    try:

        response = llm.invoke(prompt)

        content = response.content

        if isinstance(content, list):

            parts = []

            for item in content:

                if isinstance(item, dict):

                    parts.append(
                        str(item.get("text", ""))
                    )

                else:

                    parts.append(str(item))

            content = "\n".join(parts)

        return {
            "final_report": str(content).strip()
        }

    except Exception as error:

        logger.exception("Lead Agent failed")

        return {
            "final_report": (
                "# Dev Crew Final Report\n\n"
                "The multi-agent workflow completed, "
                "but the Lead Agent could not generate "
                "the final synthesis.\n\n"
                f"Error: {error}"
            )
        }


# ============================================================
# LANGGRAPH
# ============================================================

builder = StateGraph(DevCrewState)


builder.add_node(
    "requirements_agent",
    requirements_agent
)

builder.add_node(
    "architect_agent",
    architect_agent
)

builder.add_node(
    "developer_agent",
    developer_agent
)

builder.add_node(
    "testing_agent",
    testing_agent
)

builder.add_node(
    "reviewer_agent",
    reviewer_agent
)

builder.add_node(
    "lead_agent",
    lead_agent
)


# ============================================================
# WORKFLOW
# ============================================================

builder.set_entry_point(
    "requirements_agent"
)

builder.add_edge(
    "requirements_agent",
    "architect_agent"
)

builder.add_edge(
    "architect_agent",
    "developer_agent"
)

builder.add_edge(
    "developer_agent",
    "testing_agent"
)

builder.add_edge(
    "testing_agent",
    "reviewer_agent"
)

builder.add_edge(
    "reviewer_agent",
    "lead_agent"
)

builder.add_edge(
    "lead_agent",
    END
)


dev_crew_graph = builder.compile()


# ============================================================
# LANGSERVE
# ============================================================

def run_dev_crew(
    user_request: str
) -> str:

    if not isinstance(user_request, str):

        user_request = str(user_request)

    user_request = user_request.strip()

    if not user_request:

        return "Please enter a software development request."

    logger.info(
        "Dev Crew request: %s",
        user_request
    )

    try:

        result = dev_crew_graph.invoke(
            {
                "user_request": user_request
            }
        )

        return result.get(
            "final_report",
            "Dev Crew completed without a final report."
        )

    except Exception as error:

        logger.exception(
            "Dev Crew workflow failed"
        )

        return (
            "# Dev Crew Error\n\n"
            f"{error}"
        )


agent = RunnableLambda(
    run_dev_crew
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Dev Crew - LangGraph Multi-Agent",
    version="2.0.0",
    description=(
        "Multi-agent software development team "
        "implemented with LangGraph."
    )
)


# ============================================================
# LANGSERVE ROUTE
# ============================================================

add_routes(
    app,
    agent,
    path="/agent",
    input_type=str,
    output_type=str,
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "agent": "Dev Crew",
        "framework": "LangGraph",
        "type": "Multi-Agent",
        "status": "running",
        "agents": [
            "Requirements Agent",
            "Architect Agent",
            "Developer Agent",
            "Testing Agent",
            "Reviewer Agent",
            "Lead Agent"
        ],
        "playground": "/agent/playground/"
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "agent": "Dev Crew",
        "framework": "LangGraph"
    }


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                "8000"
            )
        )
    )
