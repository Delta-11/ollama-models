import dspy
from typing import Literal

lm = dspy.LM('ollama_chat/deepseek-r1:1.5b', api_base='http://localhost:11434', api_key='')
dspy.configure(lm=lm)

response = lm("How many number of R is there in word Strawberry?")
print("Response: ", response)
# print(len(lm.history))
# print(lm.history)


def search(query: str) -> list[str]:
    results = dspy.ColBERTv2(url='./README.md')(query, k=3)
    return [x['text'] for x in results]

rag = dspy.ChainOfThought('context, question -> response')

question = "What's the introduction of this book?"
response2 =rag(context=search(question), question=question)
print("Response: ", response2)



class Classify(dspy.Signature):
    """Classify sentiment of a given sentence."""

    sentence: str = dspy.InputField()
    sentiment: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    confidence: float = dspy.OutputField()

classify = dspy.Predict(Classify)
classify(sentence="This book was super fun to read, though not the last chapter.")