from llama2_for_langchain import Llama2

# 这里以调用FlagAlpha/Atom-7B-Chat为例
llm = Llama2(model_name_or_path='/home/chwu/MODELS/Llama3-Chinese-8B-Instruct')

while True:
    human_input = input("Human: ")
    response = llm(human_input)
    print(f"Llama2: {response}")