import argparse
import os

from dotenv import load_dotenv
from langchain.chains import LLMChain, SequentialChain
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
  input_variables=["language", "task"],
  template="Write a very short {language} function that will {task}",
)
test_prompt = PromptTemplate(
  input_variables=["language", "code"],
  template="Write a test for the following {language} code:\n{task}",
)

code_chain = LLMChain(
  llm=llm,
  prompt=code_prompt,
  output_key="code",
)
test_chain = LLMChain(
  llm=llm,
  prompt=test_prompt,
  output_key="test",
)

chain = SequentialChain(
  chains=[code_chain, test_chain],
  input_variables=["task", "language"],
  output_variables=["test", "code"],
)

result = chain(
  {
    "language": args.language,
    "task": args.task,
  }
)

print(">>>>>Generated Code")
print(result["code"])
print("\n\n")
print(">>>>>Generated Test")
print(result["test"])
