FROM python:3.9-slim

WORKDIR /app

# Copy only the Streamlit app and its requirements
COPY streamlit_app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code
COPY streamlit_app/ .


EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]