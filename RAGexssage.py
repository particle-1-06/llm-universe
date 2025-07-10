from langchain_openai import ChatOpenAI
# 这里我们将参数temperature设置为0.0，从而减少生成答案的随机性。
# 如果你想要每次得到不一样的有新意的答案，可以尝试调整该参数。
llm = ChatOpenAI(temperature=0, openai_api_key="sk-jYpyX1XfWIPKGTbaxunMyHK35MEa2HSpFloVw9sRUNU774Os", base_url="https://xiaoai.plus/v1")

from langchain_core.prompts import ChatPromptTemplate

template = "你是一个翻译助手，可以帮助我将 {input_language} 翻译成 {output_language}."
human_template = "{text}"

chat_prompt = ChatPromptTemplate([
    ("system", template),
    ("human", human_template),
])

text = "我带着比身体重的行李，\
游入尼罗河底，\
经过几道闪电 看到一堆光圈，\
不确定是不是这里。\
"
messages  = chat_prompt.invoke({"input_language": "中文", "output_language": "英文", "text": text})
# print(messages)

output  = llm.invoke(messages)
# print(output)

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()
print(output_parser.invoke(output))

chain = chat_prompt | llm | output_parser
print(chain.invoke({"input_language":"中文", "output_language":"英文","text": text}))

text = 'I carried luggage heavier than my body and dived into the bottom of the Nile River. After passing through several flashes of lightning, I saw a pile of halos, not sure if this is the place.'
print(chain.invoke({"input_language": "英文", "output_language": "中文","text": text}))


