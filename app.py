import streamlit as st
import dashscope
from dashscope import Generation
import PyPDF2
import io
import re

# -------------------------- 配置区域 --------------------------
# 请在此处填入你的 DashScope API Key
DASHSCOPE_API_KEY = "sk-74375db1e2554097bb3bb990275fb618"
dashscope.api_key = DASHSCOPE_API_KEY

# 金融专业风格配色
PRIMARY_COLOR = "#0a1628"
SECONDARY_COLOR = "#1e3a5f"
ACCENT_COLOR = "#4a90d9"
SUCCESS_COLOR = "#28a745"
WARNING_COLOR = "#ffc107"
DANGER_COLOR = "#dc3545"

# -------------------------- 初始化会话状态 --------------------------
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "focus_sectors" not in st.session_state:
        st.session_state.focus_sectors = []
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "current_mode" not in st.session_state:
        st.session_state.current_mode = "normal"  # normal, gold_compress, debate
    if "processing_steps" not in st.session_state:
        st.session_state.processing_steps = []
    if "sentiment_score" not in st.session_state:
        st.session_state.sentiment_score = 50

# -------------------------- 文件处理函数 --------------------------
def extract_text_from_pdf(pdf_file):
    """从 PDF 文件中提取文本"""
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        st.error(f"PDF 解析错误: {str(e)}")
    return text

def extract_text_from_txt(txt_file):
    """从 TXT 文件中提取文本"""
    try:
        return txt_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"TXT 解析错误: {str(e)}")
        return ""

def get_uploaded_content():
    """获取所有上传文件的内容"""
    all_content = ""
    for filename, content in st.session_state.uploaded_files.items():
        all_content += f"【{filename}】\n{content}\n\n"
    return all_content

# -------------------------- 生成 System Prompt --------------------------
def build_system_prompt():
    """根据当前模式和关注赛道构建系统提示词"""
    base_prompt = "你是一位资深的金融分析师，精通宏观经济、行业研究和投资分析。"
    
    # 添加关注赛道
    if st.session_state.focus_sectors:
        sectors_str = "、".join(st.session_state.focus_sectors)
        base_prompt += f"用户当前关注的赛道包括：{sectors_str}。回答时请优先考虑这些领域。"
    
    # 根据模式调整提示词
    if st.session_state.current_mode == "gold_compress":
        base_prompt += """
你现在处于「黄金压缩模式」。请将回答严格控制在150字以内，并按照以下结构输出：
【核心逻辑】：简要说明核心观点和逻辑框架
【预期差】：指出市场可能存在的预期偏差
【催化剂】：列出可能影响走势的关键事件或数据
"""
    elif st.session_state.current_mode == "debate":
        base_prompt += """
你现在处于「辩论对抗模式」。请扮演一位挑剔的首席风险官（CRO），专门寻找用户观点中的漏洞和潜在风险。
请从以下角度进行批判性分析：
1. 数据可靠性：质疑数据来源和统计方法
2. 逻辑漏洞：寻找论证中的矛盾和不严谨之处
3. 风险因素：识别未被充分考虑的风险点
4. 替代假设：提出与用户观点相反的合理假设
请用专业但尖锐的语言指出问题所在。
"""
    
    return base_prompt

# -------------------------- API 调用函数 --------------------------
def call_dashscope(messages):
    """调用 DashScope API 生成回复"""
    try:
        # 设置API Key
        dashscope.api_key = DASHSCOPE_API_KEY
        
        # 使用 Generation API
        response = Generation.call(
            model="qwen-max",
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )
        
        # 检查响应状态
        if response is None:
            return "API 调用失败: 响应为空"
        
        # 检查状态码
        if hasattr(response, 'status_code') and response.status_code != 200:
            error_msg = getattr(response, 'message', '未知错误')
            return f"API 调用失败 [状态码: {response.status_code}]: {error_msg}"
        
        # 尝试解析响应 - DashScope Generation API 返回格式为 response.output.text
        if hasattr(response, 'output') and response.output is not None:
            # 格式1: response.output.text (主要格式)
            if hasattr(response.output, 'text') and response.output.text:
                return response.output.text
            
            # 格式2: response.output.choices[0].message.content
            if hasattr(response.output, 'choices') and response.output.choices:
                if len(response.output.choices) > 0:
                    choice = response.output.choices[0]
                    if hasattr(choice, 'message') and choice.message is not None:
                        if hasattr(choice.message, 'content') and choice.message.content:
                            return choice.message.content
        
        # 尝试直接获取text属性
        if hasattr(response, 'text') and response.text:
            return response.text
        
        # 返回错误信息
        error_info = ""
        if hasattr(response, 'message'):
            error_info = response.message
        elif hasattr(response, 'error'):
            error_info = str(response.error)
        elif isinstance(response, dict) and 'error' in response:
            error_info = str(response['error'])
        
        return f"API 调用失败: {error_info if error_info else '未知错误格式'}"
        
    except Exception as e:
        return f"调用错误: {str(e)}"

# -------------------------- 情绪分析函数 --------------------------
def analyze_sentiment(text):
    """分析文本情绪，返回 0-100 的分数"""
    prompt = f"""请分析以下金融文本的情绪倾向，返回一个 0-100 的分数。
0 表示极度看空，50 表示中性，100 表示极度看多。

文本：{text[:500]}

只返回数字，不要其他内容。"""
    
    messages = [{"role": "user", "content": prompt}]
    result = call_dashscope(messages)
    
    try:
        score = int(re.search(r'\d+', result).group())
        return max(0, min(100, score))
    except:
        return 50

# -------------------------- UI 组件 --------------------------
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown(f"<h3 style='color:{ACCENT_COLOR}'>⚙️ 系统设置</h3>", unsafe_allow_html=True)
        
        # 模式切换
        st.markdown("---")
        st.markdown("**专业模式切换**")
        mode = st.radio(
            "选择工作模式",
            ["普通对话", "黄金压缩模式", "辩论对抗模式"],
            index=["normal", "gold_compress", "debate"].index(st.session_state.current_mode)
        )
        st.session_state.current_mode = {"普通对话": "normal", "黄金压缩模式": "gold_compress", "辩论对抗模式": "debate"}[mode]
        
        # 关注赛道管理
        st.markdown("---")
        st.markdown("**关注赛道管理**")
        new_sector = st.text_input("添加关注赛道")
        if st.button("添加") and new_sector.strip():
            if new_sector.strip() not in st.session_state.focus_sectors:
                st.session_state.focus_sectors.append(new_sector.strip())
        
        if st.session_state.focus_sectors:
            st.markdown("**当前关注：**")
            sectors_to_remove = []
            for idx, sector in enumerate(st.session_state.focus_sectors):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"{idx+1}. {sector}")
                with col2:
                    if st.button("×", key=f"del_{idx}"):
                        sectors_to_remove.append(idx)
            
            # 从后往前删除以避免索引混乱
            for idx in reversed(sectors_to_remove):
                st.session_state.focus_sectors.pop(idx)
        
        # 文件上传
        st.markdown("---")
        st.markdown("**文件上传（RAG）**")
        uploaded_files = st.file_uploader(
            "上传 PDF 或 TXT 文件",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in st.session_state.uploaded_files:
                    if file.type == "application/pdf":
                        content = extract_text_from_pdf(file)
                    else:
                        content = extract_text_from_txt(file)
                    st.session_state.uploaded_files[file.name] = content
                    st.success(f"已解析：{file.name}")
        
        if st.session_state.uploaded_files:
            st.markdown("**已上传文件：**")
            for filename in st.session_state.uploaded_files.keys():
                st.write(f"- {filename}")
            if st.button("清除所有文件"):
                st.session_state.uploaded_files = {}

def render_sentiment_thermometer():
    """渲染情绪温度计"""
    st.markdown(f"<h3 style='color:{ACCENT_COLOR}'>📊 市场情绪温度计</h3>", unsafe_allow_html=True)
    col1, col2 = st.columns([4, 1])
    with col1:
        st.progress(st.session_state.sentiment_score / 100)
    with col2:
        st.metric("情绪指数", f"{st.session_state.sentiment_score}%")
    
    # 情绪标签
    sentiment_label = "中性"
    if st.session_state.sentiment_score >= 70:
        sentiment_label = "看多"
        color = SUCCESS_COLOR
    elif st.session_state.sentiment_score <= 30:
        sentiment_label = "看空"
        color = DANGER_COLOR
    else:
        color = WARNING_COLOR
    
    st.markdown(f"<p style='text-align:center;color:{color};font-weight:bold'>当前情绪：{sentiment_label}</p>", unsafe_allow_html=True)

def render_messages():
    """渲染对话历史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# -------------------------- 主应用逻辑 --------------------------
def main():
    init_session_state()
    
    # 页面配置
    st.set_page_config(
        page_title="金融研报深度解析 Agent",
        page_icon="📈",
        layout="wide"
    )
    
    # 自定义样式
    st.markdown(f"""
    <style>
    .stApp {{
        background-color: {PRIMARY_COLOR};
    }}
    .stSidebar {{
        background-color: {SECONDARY_COLOR};
    }}
    .stChatMessage {{
        background-color: {SECONDARY_COLOR};
        border-radius: 10px;
        padding: 12px;
    }}
    .stButton>button {{
        background-color: {ACCENT_COLOR};
        color: white;
        border-radius: 8px;
    }}
    .stProgress > div > div {{
        background-color: {ACCENT_COLOR};
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # 标题
    st.markdown(f"""
    <h1 style='text-align:center;color:{ACCENT_COLOR};font-size:32px'>
        📊 金融研报深度解析 Agent
    </h1>
    <p style='text-align:center;color:#8892a6'>
        专业级金融分析助手 | 支持黄金压缩模式与辩论对抗模式
    </p>
    """, unsafe_allow_html=True)
    
    # 情绪温度计
    render_sentiment_thermometer()
    
    # 侧边栏
    render_sidebar()
    
    # 对话区域
    st.markdown("---")
    st.markdown(f"<h3 style='color:{ACCENT_COLOR}'>💬 对话区域</h3>", unsafe_allow_html=True)
    
    # 渲染历史消息
    render_messages()
    
    # 用户输入
    if prompt := st.chat_input("请输入您的问题或观点..."):
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 构建完整消息
        messages = [{"role": "system", "content": build_system_prompt()}]
        
        # 添加上传文件内容
        uploaded_content = get_uploaded_content()
        if uploaded_content:
            messages.append({"role": "user", "content": f"参考文档：\n{uploaded_content}"})
        
        # 添加对话历史
        messages.extend(st.session_state.messages)
        
        # 显示处理状态
        processing_steps = [
            "识别用户意图...",
            "检索本地知识库...",
            "分析核心逻辑...",
            "生成专业回复..."
        ]
        
        if st.session_state.current_mode == "debate":
            processing_steps.insert(2, "逻辑对抗分析...")
        
        with st.status("📋 分析处理中...", expanded=True) as status:
            for step in processing_steps:
                st.write(f"🔄 {step}")
            
            # 调用 API
            with st.spinner("AI 正在分析..."):
                response = call_dashscope(messages)
            
            status.update(label="✅ 分析完成", state="complete", expanded=False)
        
        # 更新情绪分数
        st.session_state.sentiment_score = analyze_sentiment(response)
        
        # 添加助手回复
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)
        
        # 刷新页面
        st.rerun()

if __name__ == "__main__":
    main()