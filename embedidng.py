from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your api key") 
response = client.embeddings.create(
    model="embedding-3", #填写需要调用的模型编码
     input=[
        "美食非常美味，服务员也很友好。",
        "这部电影既刺激又令人兴奋。",
        "阅读书籍是扩展知识的好方法。"
    ],
)
print(response)

{
    "model": "embedding-3",
    "data": [
        {
            "embedding": [
                -0.02675454691052437,
                0.019060475751757622,
                ...... 
                -0.005519774276763201,
                0.014949671924114227
            ],
            "index": 0,
            "object": "embedding"
        },
        ...
        {
            "embedding": [
                -0.02675454691052437,
                0.019060475751757622,
                ...... 
                -0.005519774276763201,
                0.014949671924114227
            ],
            "index": 2,
            "object": "embedding"
        }
    ],
    "object": "list",
    "usage": {
        "completion_tokens": 0,
        "prompt_tokens": 100,
        "total_tokens": 100
    }
}