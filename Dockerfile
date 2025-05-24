# استخدام صورة أساسية خفيفة مع بايثون 3.11
FROM python:3.11-slim

# تعيين مجلد العمل
WORKDIR /app

# نسخ ملفات المشروع
COPY . .

# تثبيت المتطلبات
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# الأمر التشغيلي
CMD ["streamlit", "run", "WebApp.py", "--server.port=8501", "--server.address=0.0.0.0"]
