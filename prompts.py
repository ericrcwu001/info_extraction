#!/usr/bin/python3

from langchain_core.prompts.prompt import PromptTemplate

def summarize_template(tokenizer, additional_instructions = None, accumulated = False):
  messages = [
    {'role': 'system', 'content': 'Rewrite this text in summarized form.' + ("" if additional_instructions is None else f"\n\n{additional_instructions}")},
    {'role': 'user', 'content': "{chunk}" if accumulated == False else "Previous summaries:\n\n{accumulated_summaries_string}\n\n{chunk}"}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['chunk'] if accumulated == False else ['chunk', 'accumulated_summaries_string'])
  return template

def rag_template(tokenizer):
  messages = [
    {'role': 'system', 'content': 'Use the given context to answer the question. If the context doesn\'t give you a clue to the answer, just say you don\'t know and don\'t try to make up an answer. Try to keep your answer brief and to the point.'},
    {'role': 'user', 'content': 'context: {context}\nquestion:{question}\nanswer:'}
  ]
  prompt = tokenizer.apply_chat_template(messages, tokenize = False, add_generating_prompt = True)
  template = PromptTemplate(template = prompt, input_variables = ['context', 'question'])
  return template

