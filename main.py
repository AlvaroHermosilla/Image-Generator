import os
from typing import TypedDict, Annotated, Sequence, List
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from openai import OpenAI
import asyncio
import nodes
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
# Define the state dictionary.
class State (TypedDict):
    query:str
    plot: str
    character_description: str
    image_prompt: List[str]
    image_urls: List[str]

# Instantiate OpenAI
llm = ChatOpenAI(model = "gpt-4", temperature = 1 )
client = OpenAI()

#Define the workflow.

workflow = StateGraph(State)  

generate_character_description_node = nodes.GenerateCharacterDescription(llm=llm)
generate_plot_description_node = nodes.GeneratePlotDescription(llm=llm)
generate_image_prompt_node = nodes.GenerateImagePrompts(llm=llm)
create_image_node = nodes.CreateImage(client=client)

workflow.add_node("character_description", generate_character_description_node.generate_character_description)
workflow.add_node("plot_description", generate_plot_description_node.generate_plot_description)
workflow.add_node("image_prompt", generate_image_prompt_node.generate_image_prompt)
workflow.add_node("create_image", create_image_node.create_and_save_image)

workflow.add_edge("character_description", "plot_description")
workflow.add_edge("plot_description", "image_prompt")
workflow.add_edge("image_prompt", "create_image")
workflow.add_edge("create_image", END)

workflow.set_entry_point("character_description")
app = workflow.compile()

# Define the run function
async def run_workflow(query: str):
    initial_state = {
        "messages": [],
        "query": query,
        "plot": "",
        "character_description": "",
        "image_prompt": [],
        "image_urls": []
    }

    result = await app.ainvoke(initial_state)

    print("Character/Scene Description:")
    print(result["character_description"])

    print("\nGenerated Plot:")
    print(result["plot"])

    print("\nImage Prompts:")
    for i, prompt in enumerate(result["image_prompt"], 1):
        print(f"{i}. {prompt}")

    return result

# Execute
query = "A dog playing with a little ball on a small hill."
result = asyncio.run(run_workflow(query))
