import os
from ..search import TellGPT, llm, config, conversation_stages
import json
import sys
import time
import streamlit as st
import tiktoken
from deepseek import ChatDeepSeek
from tell_server import tell_agent

@st.cache_resource
def get_openai_client(model_name, api_key,temperature):
    #使用了缓存，当参数不变时，不会重复创建client
    llm = ChatDeepSeek(model_name=model_name,api_key=api_key,temperature=temperature)
    return llm


def chat_page():
    st.title("🌟 Machine Learning Agent ")

    if 'number1' not in st.session_state:
        st.session_state['number1'] = '2410043017  颜东洋'
    if 'number2' not in st.session_state:
        st.session_state['number2'] = '2410043008  徐梓浩'
    if 'number3' not in st.session_state:
        st.session_state['number3'] = '2410043055  谢杰钊'
    st.session_state.number1 = st.sidebar.text_input('Group members',st.session_state.number1)
    st.session_state.number2 = st.sidebar.text_input('Group members',st.session_state.number2)
    st.session_state.number3 = st.sidebar.text_input('Group members',st.session_state.number3)


    src_path = os.path.dirname(os.path.realpath(sys.argv[0]))
    # 读取默认配置文件
    with open(os.path.join(src_path, 'config/default.json'), 'r', encoding='utf-8') as f:
        config_defalut = json.load(f)

    st.session_state['model_list'] = config_defalut["completions"]["models"]
    model_name = st.selectbox('Models', st.session_state.model_list,)
    if not st.checkbox('default param', True):
        max_tokens = st.number_input('Max Tokens', 1, 200000, config_defalut["completions"]["max_tokens"],
                                     key='max_tokens')
        temperature = st.slider('Temperature', 0.0, 1.0, config_defalut["completions"]["temperature"],
                                key='temperature')
        top_p = st.slider('Top P', 0.0, 1.0, config_defalut["completions"]["top_p"], key='top_p')
        stream = st.checkbox('Stream', config_defalut["completions"]["stream"], key='stream')
    else:
        max_tokens = config_defalut["completions"]["max_tokens"]
        temperature = config_defalut["completions"]["temperature"]
        top_p = config_defalut["completions"]["top_p"]
        stream = config_defalut["completions"]["stream"]

    if model_name == 'deepseek-reasoner':
        llm = get_openai_client(model_name='deepseek-reasoner', api_key='sk-79c95513a6f74dd9b56bf9dbfdf1dae0',
                            temperature=temperature)
    


    # 初始化智能体
    if "tell_agent" not in st.session_state:

        tell_agent = TellGPT.from_llm(llm, verbose=False, **config)
        if os.path.exists("history.json"):
            with open("history.json", "r", encoding="utf-8") as f:
                tell_agent.conversation_history = json.load(f)
        else:
            tell_agent.seed_agent()  # 没历史就重置

        st.session_state.tell_agent = tell_agent
    else:
        tell_agent = st.session_state.tell_agent

    if "chat_messages" not in st.session_state:
        chat_messages = []

        with open(os.path.join(src_path, 'config/history.json'), 'r', encoding='utf-8') as f:
            history = json.load(f)
            for item in history:
                if isinstance(item, str) and ":" in item:
                    role, content = item.split(":", 1)
                    role = "user" if "User" in role else "assistant"
                    chat_messages.append({"role": role, "content": content.replace("<END_OF_TURN>", "").strip()})
        st.session_state['chat_messages'] = chat_messages

        # 清除历史记录按钮
    if st.button("🧹 清除对话"):
        st.session_state.chat_messages = []
        st.session_state.tell_agent.seed_agent()
        if os.path.exists("history.json"):
            os.remove("history.json")

        # 显示历史消息
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

        # 获取用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        tell_agent.human_step(prompt)
        st.session_state.chat_messages.append({"role": "user", "content": prompt})

        # 显示 AI 回复
        with st.chat_message("assistant"):
            with st.spinner("小D思考中..."):
                tell_agent.determine_conversation_stage()
                tell_agent.step()  # 内部自动更新 conversation_history
                with open(os.path.join(src_path, 'config/history.json'), "w", encoding="utf-8") as f:
                    json.dump(tell_agent.conversation_history, f, ensure_ascii=False, indent=2)


                # 获取最新回复
                latest_reply = tell_agent.conversation_history[-1]
                content = latest_reply.split(":", 1)[-1].replace("<END_OF_TURN>", "").strip()

                st.markdown(content)
                st.session_state.chat_messages.append({"role": "assistant", "content": content})







def num_tokens_from_messages(messages, model):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
        }:
        # 每个消息有一个基本的令牌数 tokens_per_message，默认3个token，每个 name 属性预设的固定令牌数 tokens_per_name，假设其值为 1。
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    #函数通过迭代消息列表，并根据消息的角色 (如 user、assistant、tool、system) 计算令牌数量。
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # 每个回复都以 <|start|>assistant<|message|> 开头
    # 例如：<|start|>assistant<|message|>今天天气很好，适合出门！ <|end|>
    num_tokens += 3
    return num_tokens

if __name__ == "__main__":
    chat_page()
