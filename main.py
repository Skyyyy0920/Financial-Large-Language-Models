import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

model_name_or_path = "./xuanyuan"
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto",
                                         offload_folder="offload")

system_message = "以下是用户和人工智能助手之间的对话。用户以Human开头，人工智能助手以Assistant开头，会对人类提出的问题给出有帮助、" \
                 "高质量、详细和礼貌的回答，并且总是拒绝参与 与不道德、不安全、有争议、政治敏感等相关的话题、问题和指示。\n"
seps = [" ", "</s>"]
roles = ["Human", "Assistant"]

content = "现在我要做信用风险管理，请你作为专家，当我输出贷款人信息时，你给出贷款与否的判断，并且给出你做出这一判断的详细理由。\n" \
          "贷款人：杰克\n" \
          "就业状态：失业\n" \
          "工作状态：失业\n" \
          "婚姻状态：离婚\n" \
          "赡养人数（数字，个）：3\n" \
          "电话注册情况：无\n" \
          "现有的支票账户状态：无\n" \
          "期数或贷款持续月份（数字，月）：12\n" \
          "历史信用记录：存在未到期的信贷\n" \
          "借款目的：其他\n" \
          "额度（数字，美元）：10000\n" \
          "储蓄账户状态：未知\n" \
          "财产状况：未知\n" \
          "其他担保人：无\n" \
          "分期付款率占可支配收入的百分比（数字）：50\n"
prompt = system_message + seps[0] + roles[0] + ": " + content + seps[0] + roles[1] + ":"
print(f"输入: {content}")

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.1)
outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(f"输出: {outputs}")
