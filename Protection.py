import re
import time
import hashlib
import random
import string

_rate_limit_store = {}
_captcha_store = {}
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX_REQ = 15

def sanitize_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r'[\u200B-\u200D\uFEFF]', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()

def validate_request(text: str, max_chars: int = 20000) -> tuple[bool, str]:
    if not text or len(text) < 10: return False, "Text too short."
    if len(text) > max_chars: return False, "Text exceeds maximum length."
    return True, "OK"

def check_rate_limit(ip: str) -> bool:
    now = time.time()
    
    # Clean up expired rate limits to prevent memory leak
    if len(_rate_limit_store) > 1000:
        for k in list(_rate_limit_store.keys()):
            _rate_limit_store[k] = [t for t in _rate_limit_store[k] if now - t < RATE_LIMIT_WINDOW]
            if not _rate_limit_store[k]:
                del _rate_limit_store[k]

    if ip not in _rate_limit_store: _rate_limit_store[ip] = []
    _rate_limit_store[ip] = [t for t in _rate_limit_store[ip] if now - t < RATE_LIMIT_WINDOW]
    
    if len(_rate_limit_store[ip]) >= RATE_LIMIT_MAX_REQ: return False
    _rate_limit_store[ip].append(now)
    return True

def get_security_headers():
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        # Allowed blob: for the Web Worker Captcha
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline' blob:; style-src 'self' 'unsafe-inline'; worker-src 'self' blob:;"
    }

def generate_challenge():
    challenge = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
    _captcha_store[challenge] = time.time() + 300
    return challenge

def verify_captcha(challenge: str, nonce: str) -> bool:
    if challenge not in _captcha_store: return False
    if _captcha_store[challenge] < time.time():
        del _captcha_store[challenge]
        return False
    
    test_hash = hashlib.sha256((challenge + nonce).encode()).hexdigest()
    if test_hash.startswith("0000"):
        del _captcha_store[challenge]
        return True
    return False