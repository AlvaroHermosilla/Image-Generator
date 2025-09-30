from langchain_core.prompts import ChatPromptTemplate
import asyncio
import aiohttp
from PIL import Image
import io
from typing import TypedDict, List

class State(TypedDict):
    query: str
    plot: str
    character_description: str
    image_prompt: List[str]  # singular
    image_urls: List[str]
    
class GenerateCharacterDescription:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_character_description(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant that generates a detailed description related to a query about the main character."),
            ("human", "Based on the query:{query}, create a detailed description of the main character, object, or scene. "
                      "Include specific details about appearance, characteristics, and any unique features. "
                      "The description should be detailed enough for image generation.")
        ])
        chain = prompt | self.llm
        character_description = chain.invoke({"query": state["query"]}).content
        return {**state, "character_description": character_description}
    
class GeneratePlotDescription:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_plot_description(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system","You are an assistant that generates a detailed plot description"),
            ("human", "Based on the query:{query} and featuring this description: {character_description}, create a detailed description of a single scene for an image. Include appearance, setting, atmosphere, and any unique features.")
        ])
        chain = prompt | self.llm
        plot_description = chain.invoke({
            "query": state["query"],
            "character_description": state["character_description"]
        }).content
        return {**state, "plot": plot_description}
    
class GenerateImagePrompts:
    def __init__(self, llm):
        self.llm = llm
    
    def generate_image_prompt(self, state: State) -> State:
        prompt = ChatPromptTemplate.from_messages([
            ("system","You are an assistant that generates detailed prompts"),
            ("human", "Based on the plot:{plot} and featuring this description {character_description}, generate a detailed family-friendly prompt suitable for DALL-E.")
        ])
        chain = prompt | self.llm
        image_prompt = chain.invoke({
            "plot": state["plot"],
            "character_description": state["character_description"]
        }).content
        return {**state, "image_prompt": state["image_prompt"] + [image_prompt]}
    
class CreateImage:
    def __init__(self, client):
        self.client = client

    async def create_and_save_image(self, state: State, filename: str = "output.png") -> State:
        """Genera la imagen a partir del último prompt y la guarda como archivo."""
        if not state["image_prompt"]:
            raise ValueError("No se encontró 'image_prompt' en el state")
        
        prompt = state["image_prompt"][-1]
        url = await self._generate_image(prompt)
        await self._download_image(url, filename)

        return {**state, "image_urls": state["image_urls"] + [url]}

    async def _generate_image(self, prompt: str) -> str:
        response = self.client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        return response.data[0].url

    async def _download_image(self, url: str, filename: str):
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                img_bytes = await resp.read()
                with open(filename, "wb") as f:
                    f.write(img_bytes)
        print(f"Imagen guardada en {filename}")
