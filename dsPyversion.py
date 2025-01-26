import dspy
from typing import Literal

lm = dspy.LM('ollama_chat/deepseek-r1:1.5b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

# Normal Prompting
response = lm("How many number of R is there in word Strawberry?")
print("Response: ", response)
# print(len(lm.history))
# print(lm.history)

# THis is RAG search with colbert
def search(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='http://20.102.90.50:2017/wiki17_abstracts')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What's the name of the castle that David Gregory inherited?"

rag(context=search(question), question=question)


# This is for sentiment analysis
questionsList = ["Are you out of your mind?", "He didn't deliver on his part.", "She gave a great job talk at the conference.", "The project was challenging."]

class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
for sentence in questionsList:
    response = classify(sentence=sentence)
    print(f"Raw Response for {sentence}: {response}")
    print(f"Response for {sentence}: {response}\n")



class ExtractInfo(dspy.Signature):
    """Extract structured information from text."""

    text: str = dspy.InputField()
    knowledge: str = dspy.OutputField()
    diverse: list[str] = dspy.OutputField()
    impact: list[str] = dspy.OutputField()
    entities: list[dict[str, str]] = dspy.OutputField(desc="a list of entities and their metadata")

module = dspy.Predict(ExtractInfo)

text = "Whether it's a scientific breakthrough, a thought-provoking piece of art, or the latest technological innovation, there is always something fascinating to explore. With each passing day, we uncover new discoveries and push the boundaries of human knowledge. From the intricacies of the natural world to the vast depths of outer space, every subject offers a unique opportunity for exploration and understanding. Through curiosity and perseverance, we continue to expand our horizons and shape the world we live in. The possibilities are endless, and the journey of discovery never ceases."
response = module(text=text)

print(response.knowledge)
print(response.diverse)
print(response.impact)
print(response.entities)