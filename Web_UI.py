import gradio as gr


# 定义一个简单的函数，这个函数会将输入的文本加上感叹号并返回
def add_exclamation(text):
    return text + "!"


# 创建一个文本框输入组件
text_input = gr.Textbox(lines=2, label="输入一些文本")

# 创建一个输出组件
output_text = gr.Textbox(label="结果")

# 创建 Gradio 接口
interface = gr.Interface(
    fn=add_exclamation,
    inputs=text_input,
    outputs=output_text,
    title="加感叹号",
    description="这个演示会将输入的文本加上感叹号。",
    theme="compact"
)

# 设置端口为5000
interface.launch(share=True, server_port=5001)
print("ok")
