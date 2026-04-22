import dashscope
from dashscope import Generation

# 设置API Key
dashscope.api_key = "sk-74375db1e2554097bb3bb990275fb618"

try:
    response = Generation.call(
        model="qwen-max",
        messages=[{"role": "user", "content": "你好"}]
    )
    
    print("响应类型:", type(response))
    print("响应内容:", response)
    print("\n状态码:", getattr(response, 'status_code', '无'))
    
    # 尝试解析响应
    if hasattr(response, 'output'):
        print("\noutput:", response.output)
        if hasattr(response.output, 'choices'):
            print("choices:", response.output.choices)
            if response.output.choices:
                choice = response.output.choices[0]
                print("choice:", choice)
                if hasattr(choice, 'message'):
                    print("message:", choice.message)
                    if hasattr(choice.message, 'content'):
                        print("content:", choice.message.content)

except Exception as e:
    print("错误:", str(e))
    import traceback
    traceback.print_exc()