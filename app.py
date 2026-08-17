"""
FastAPI web app that uses llm.py to talk to configured LLM provider.
"""

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

from llm import get_llm

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize LLM client once at startup
llm_client = None


@app.on_event("startup")
async def startup_event():
    global llm_client
    try:
        llm_client = get_llm()
    except Exception as e:
        print(f"Warning: Could not initialize LLM on startup: {e}")


@app.get("/", response_class=HTMLResponse)
async def main(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": "",
        "llm_response": "",
        "error": ""
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

        response = llm_client.invoke(user_input)
        llm_response = response.content
    except Exception as e:
        error = str(e)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "llm_response": llm_response,
        "error": error
    })


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6822)
