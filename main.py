import os
import preprocess
from groq import Groq

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


client = Groq(
    api_key = os.getenv("API_KEY")
)


chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": f"job desc is {preprocess.preprocessed_job_desc}. Rank resume text present in {preprocess.preprocessed_pdf_text} in strictly descending order of their similarity with this job desc strictly and find similarity based on skills present in job desc and resume also ignore all other text",
        },

        {
            "role": "system",
            "content": "Provide a concise answer just provide the rankings"
        },

    ],
    model="llama3-70b-8192",
)

print(chat_completion.choices[0].message.content)