#!/usr/bin/python3

from os import environ
from huggingface_hub import login
from transformers import AutoTokenizer
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface import HuggingFacePipeline


def ChatGLM3(locally=False):
    login('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')
    tokenizer = AutoTokenizer.from_pretrained('THUDM/chatglm3-6b', trust_remote_code=True)
    if locally:
        llm = HuggingFacePipeline.from_model_id(
            model_id='THUDM/chatglm3-6b',
            task='text-generation',
            device=0,
            pipeline_kwargs={
                "max_length": 8192,
                "do_sample": False,
                "top_p": 0.8,
                "temperature": 0.8,
                "trust_remote_code": True,
                "use_cache": True,
                "return_full_text": False
            }
        )
    else:
        environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
        llm = HuggingFaceEndpoint(
            endpoint_url='THUDM/chatglm3-6b',
            task="text-generation",
            max_new_tokens=8192,
            do_sample=False,
            top_p=0.8,
            temperature=0.8,
            trust_remote_code=True,
            cache=True
        )
    return tokenizer, llm


def Llama2(locally=False):
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
    if locally:
        llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Llama-2-7b-chat-hf",
            task="text-generation",
            device=0,
            pipeline_kwargs={
                "max_length": 4096,
                "do_sample": False,
                "temperature": 0.8,
                "top_p": 0.8,
                "use_cache": True,
                "return_full_text": False
            }
        )
    else:
        environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
        llm = HuggingFaceEndpoint(
            endpoint_url="meta-llama/Llama-2-7b-chat-hf",
            task="text-generation",
            max_new_tokens=4096,
            do_sample=False,
            temperature=0.8,
            top_p=0.8,
            cache=True
        )
    return tokenizer, llm


def Llama3(locally=False):
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    if locally:
        llm = HuggingFacePipeline.from_model_id(
            model_id="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            device=0,
            pipeline_kwargs={
                "max_length": 16384,
                "do_sample": False,
                "temperature": 0.6,
                "top_p": 0.9,
                "eos_token_id": [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                # "use_cache": True,
                "return_full_text": False
            }
        )
    else:
        environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
        llm = HuggingFaceEndpoint(
            endpoint_url="meta-llama/Meta-Llama-3-8B-Instruct",
            task="text-generation",
            max_length=16384,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
            cache=True
        )
    return tokenizer, llm


def CodeLlama(locally=False):
    login(token='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')
    tokenizer = AutoTokenizer.from_pretrained('meta-llama/CodeLlama-7b-Instruct-hf')
    if locally:
        llm = HuggingFacePipeline.from_model_id(
            model_id='meta-llama/CodeLlama-7b-Instruct-hf',
            task='text-generation',
            device=0,
            pipeline_kwargs={
                "max_length": 16384,
                "do_sample": False,
                "temperature": 0.8,
                "top_p": 0.8,
                "use_cache": True,
                "return_full_text": False
            }
        )
    else:
        environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
        llm = HuggingFaceEndpoint(
            endpoint_url='meta-llama/CodeLlama-7b-Instruct-hf',
            task='text-generation',
            max_length=16384,
            do_sample=False,
            temperature=0.8,
            top_p=0.8,
            cache=True
        )
    return tokenizer, llm
