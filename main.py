import argparse
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="return a list of numbers")
parser.add_argument("--language", default="python")
args = parser.parse_args()

# parsing env manually for practice, but library will look for "OPENAPI_API_KEY" in env
llm = OpenAI(
  openai_api_key=os.getenv("OPENAPI_API_KEY"),
)

code_prompt = PromptTemplate(
  template="Write a very short {language} function that will {task}",
  input_variables=["language", "task"],
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
)

result = code_chain(
  {
    "language": args.language,
    "task": args.task,
  }
)

print(result["text"])

# print("test")
