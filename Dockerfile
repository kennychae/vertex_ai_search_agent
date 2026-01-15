FROM python:3.12-slim

WORKDIR /app

# 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 소스 전체
COPY . .

ENV PORT=8080
EXPOSE 8080

# ADK 서버 자동 실행
CMD ["sh", "-c", "adk api_server --host 0.0.0.0 --port ${PORT}"]