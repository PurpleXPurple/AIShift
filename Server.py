from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.concurrency import run_in_threadpool
from starlette.datastructures import MutableHeaders
import uvicorn

import Users
import Protection
import ml

BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "index.html"

@asynccontextmanager
async def lifespan(app: FastAPI):
    ml.load_pipeline()
    yield

app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

class SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        
        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for k, v in Protection.get_security_headers().items():
                    headers[k] = v
            await send(message)
            
        await self.app(scope, receive, send_wrapper)

app.add_middleware(SecurityHeadersMiddleware)

def rate_limiter(request: Request):
    ip = request.client.host if request.client else "0.0.0.0"
    if not Protection.check_rate_limit(ip):
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    if not INDEX_FILE.exists(): 
        raise HTTPException(404, "index.html missing")
    return HTMLResponse(content=INDEX_FILE.read_text(encoding="utf-8"))

@app.get("/favicon.ico")
async def favicon(): return JSONResponse({}, status_code=204)

@app.get("/api/captcha")
async def get_captcha(): return {"challenge": Protection.generate_challenge()}

@app.post("/api/detect", dependencies=[Depends(rate_limiter)])
async def detect(
    request: Request,
    text: str = Form(default=""),
    consent_dataset: bool = Form(default=False),
    model_choice: str = Form(default="finetuned"),
    captcha_challenge: str = Form(default=""),
    captcha_nonce: str = Form(default=""),
):
    try:
        if not Protection.verify_captcha(captcha_challenge, captcha_nonce):
            return JSONResponse({"detail": "Invalid or expired CAPTCHA."}, status_code=403)

        clean = Protection.sanitize_text(text)
        ok, msg = Protection.validate_request(clean)
        if not ok: return JSONResponse({"detail": msg}, status_code=400)

        result = await run_in_threadpool(ml.classify, clean, model_choice)
        ai_prob = result["score"] if result["label"].upper() == "AI" else 1.0 - result["score"]
        
        if consent_dataset:
            ip = request.client.host if request.client else "0.0.0.0"
            Users.log_to_dataset(clean, ai_prob, result["label"], ip)

        return {
            "ai_probability": round(ai_prob * 100, 2),
            "raw_label": result["label"],
            "raw_score": round(result["score"] * 100, 2),
            "char_count": len(clean),
            "model_used": model_choice
        }
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)

@app.get("/api/reviews")
async def get_reviews():
    try: return JSONResponse(Users.get_reviews())
    except Exception as e: return JSONResponse({"detail": str(e)}, status_code=500)

@app.post("/api/reviews", dependencies=[Depends(rate_limiter)])
async def post_review(
    request: Request, name: str = Form(default=""), rating: int = Form(...), text: str = Form(...),
    captcha_challenge: str = Form(default=""), captcha_nonce: str = Form(default=""),
):
    try:
        if not Protection.verify_captcha(captcha_challenge, captcha_nonce):
            return JSONResponse({"detail": "Invalid or expired CAPTCHA."}, status_code=403)

        clean = Protection.sanitize_text(text)
        if len(clean) < 5: return JSONResponse({"detail": "Review too short"}, status_code=400)
        ip = request.client.host if request.client else "0.0.0.0"
        return JSONResponse(Users.add_review(name, rating, clean, ip))
    except Exception as e:
        return JSONResponse({"detail": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run("Server:app", host="127.0.0.1", port=8000, reload=True)