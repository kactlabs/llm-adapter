"""
FastAPI web app that uses llm.py to talk to configured LLM provider.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from llm import get_llm, get_llm_info

# Initialize LLM client once at startup
llm_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm_client
    try:
        llm_client = get_llm()
    except Exception as e:
        print(f"Warning: Could not initialize LLM on startup: {e}")
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    llm_info = get_llm_info()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": "",
        "llm_response": "",
        "error": "",
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"]
    })


@app.post("/", response_class=HTMLResponse)
async def ask_llm(request: Request, user_input: str = Form("")):
    global llm_client
    llm_response = ""
    error = ""

    if not user_input.strip():
        return templates.TemplateResponse("index.html", {
            "request": request,
            "user_input": user_input,
            "llm_response": "",
            "error": "Please enter a prompt."
        })

    try:
        if llm_client is None:
            llm_client = get_llm()

        llm_info = get_llm_info()
        print(f"\n{'='*60}")
        print(f"[DEBUG] Invoking LLM: provider={llm_info['provider']}, model={llm_info['model']}")
        print(f"[DEBUG] User input: {user_input[:100]}{'...' if len(user_input) > 100 else ''}")
        print(f"[DEBUG] LLM client type: {type(llm_client).__name__}")

        response = llm_client.invoke(user_input)

        # Debug: log the full response object
        print(f"[DEBUG] Response type: {type(response).__name__}")
        print(f"[DEBUG] Response dir: {[attr for attr in dir(response) if not attr.startswith('_')]}")

        # Check for content attribute
        if hasattr(response, 'content'):
            print(f"[DEBUG] response.content type: {type(response.content)}")
            print(f"[DEBUG] response.content length: {len(response.content) if response.content else 0}")
            print(f"[DEBUG] response.content repr: {repr(response.content[:500]) if response.content else repr(response.content)}")
        else:
            print(f"[DEBUG] WARNING: response has no 'content' attribute")

        # Check for additional_kwargs (thinking models often put content here)
        if hasattr(response, 'additional_kwargs'):
            print(f"[DEBUG] response.additional_kwargs: {response.additional_kwargs}")

        # Check for response_metadata
        if hasattr(response, 'response_metadata'):
            print(f"[DEBUG] response.response_metadata: {response.response_metadata}")

        # Check for tool_calls or thinking content
        if hasattr(response, 'tool_calls'):
            print(f"[DEBUG] response.tool_calls: {response.tool_calls}")

        llm_response = response.content

        # If content is empty, try to extract from other fields
        if not llm_response:
            print(f"[DEBUG] WARNING: content is empty/None, checking alternative fields...")
            if hasattr(response, 'additional_kwargs'):
                # Some models put thinking/reasoning in additional_kwargs
                ak = response.additional_kwargs
                if 'thinking' in ak:
                    print(f"[DEBUG] Found 'thinking' in additional_kwargs (length: {len(ak['thinking'])})")
                if 'reasoning_content' in ak:
                    print(f"[DEBUG] Found 'reasoning_content' in additional_kwargs")
                    llm_response = ak['reasoning_content']
            # Try string representation as last resort
            if not llm_response:
                raw_str = str(response)
                print(f"[DEBUG] str(response): {raw_str[:500]}")

        print(f"[DEBUG] Final llm_response length: {len(llm_response) if llm_response else 0}")
        print(f"{'='*60}\n")
    except Exception as e:
        import traceback
        print(f"[DEBUG] EXCEPTION: {type(e).__name__}: {e}")
        print(f"[DEBUG] Traceback:\n{traceback.format_exc()}")
        error = str(e)

    llm_info = get_llm_info()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "llm_response": llm_response,
        "error": error,
        "llm_provider": llm_info["provider"],
        "llm_model": llm_info["model"]
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6822)
