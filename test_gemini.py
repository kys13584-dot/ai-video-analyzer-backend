import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")
print(f"[DEBUG] API Key: {str(api_key)[:10]}...")

try:
    from google import genai
    client = genai.Client(api_key=api_key)
    
    r = client.models.generate_content(
        model="models/gemini-2.0-flash",
        contents="안녕하세요! 숏폼 영상 분석 테스트입니다. 한 문장으로 반갑게 인사해주세요."
    )
    print(f"\n[SUCCESS] Gemini 응답:\n{r.text}")
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {str(e)[:500]}")
