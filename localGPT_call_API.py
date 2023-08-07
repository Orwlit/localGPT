import logging

import click
import torch

import requests
import json

# UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
# warn("The installed version of bitsandbytes was compiled without GPU support. "
# 'NoneType' object has no attribute 'cadam32bit_grad_fp32'
# from auto_gptq import AutoGPTQForCausalLM 

from huggingface_hub import hf_hub_download
from langchain.chains import RetrievalQA

from langchain.embeddings import HuggingFaceInstructEmbeddings

from langchain.llms import HuggingFacePipeline, LlamaCpp

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma


from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

# UserWarning: The installed version of bitsandbytes was compiled without GPU support. 8-bit optimizers, 8-bit multiplication, and GPU quantization are unavailable.
#   warn("The installed version of bitsandbytes was compiled without GPU support. "
# 'NoneType' object has no attribute 'cadam32bit_grad_fp32'
from transformers import LlamaForCausalLM

from transformers import LlamaTokenizer
from transformers import pipeline


from constants import CHROMA_SETTINGS, EMBEDDING_MODEL_NAME, PERSIST_DIRECTORY


def load_model(device_type, model_id, model_basename=None):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".ggml" in model_basename:
            logging.info("Using Llamacpp for GGML quantized models")
            # model_path = hf_hub_download(repo_id=model_id, filename=model_basename)
            model_path = "/Users/cr_mac_001/LLMs/llama2/Llama-2-13B-chat-GGML/llama-2-13b-chat.ggmlv3.q6_K.bin"
            max_ctx_size = 4096
            kwargs = {
                "model_path": model_path,
                "n_ctx": max_ctx_size,
                "max_tokens": max_ctx_size,
            }
            if device_type.lower() == "mps":
                kwargs["n_gpu_layers"] = 1
                kwargs["n_batch"] = 512
                kwargs["f16_kv"] = True
            if device_type.lower() == "cuda":
                kwargs["n_gpu_layers"] = 1000
                kwargs["n_batch"] = max_ctx_size
                
            return LlamaCpp(**kwargs)

        # else:
        #     # The code supports all huggingface models that ends with GPTQ and have some variation
        #     # of .no-act.order or .safetensors in their HF repo.
        #     logging.info("Using AutoGPTQForCausalLM for quantized models")

        #     if ".safetensors" in model_basename:
        #         # Remove the ".safetensors" ending if present
        #         model_basename = model_basename.replace(".safetensors", "")

        #     tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        #     logging.info("Tokenizer loaded")

        #     model = AutoGPTQForCausalLM.from_quantized(
        #         model_id,
        #         model_basename=model_basename,
        #         use_safetensors=True,
        #         trust_remote_code=True,
        #         device="cuda:0",
        #         use_triton=False,
        #         quantize_config=None,
        #     )
    elif (
        device_type.lower() == "cuda"
    ):  # The code supports all huggingface models that ends with -HF or which have a .bin
        # file in their HF repo.
        logging.info("Using AutoModelForCausalLM for full models")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logging.info("Tokenizer loaded")

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            # max_memory={0: "15GB"} # Uncomment this line with you encounter CUDA out of memory errors
        )
        model.tie_weights()
    else:
        logging.info("Using LlamaTokenizer")
        tokenizer = LlamaTokenizer.from_pretrained(model_id)
        model = LlamaForCausalLM.from_pretrained(model_id)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
def main(device_type, show_sources):
    """
    This function implements the information retrieval task.


    1. Loads an embedding model, can be HuggingFaceInstructEmbeddings or HuggingFaceEmbeddings
    2. Loads the existing vectorestore that was created by inget.py
    3. Loads the local LLM using load_model function - You can now set different LLMs.
    4. Setup the Question Answer retreival chain.
    5. Question answers.
    """

    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type}) # 不是这里弹出的bitsandbytes不支持GPU？？

    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        # client_settings=CHROMA_SETTINGS,
    )
    # retriever = db.as_retriever()

    # model_id = "TheBloke/Llama-2-13B-Chat-GGML"
    # model_basename = "llama-2-13b-chat.ggmlv3.q6_K.bin"

    # llm = load_model(device_type, model_id=model_id, model_basename=model_basename)

    # qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    # qa = RetrievalQA.from_chain_type(llm,retriever=vectorstore.as_retriever())
    # Interactive questions and answers
    


    # 设置 API 的 URL
    url = "http://localhost:11112/v1/completions"

    headers = {
        'Content-Type': 'application/json'
    }
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        # res = qa(query) # TODO: without GPU support!!!
        # answer, docs = res["result"], res["source_documents"]
        
        # initialize prompt
        prompt = f"Answer the query below using given document:\nquery: {query}\n\ndocument: \n"
        
        query_embeded = embeddings.embed_query(query)
        retrieve_result = db.similarity_search_by_vector_with_relevance_scores(query_embeded)
        for item in retrieve_result:
            doc_obj, score = item
            
            # Extract page_content and metadata from the Document object
            doc = doc_obj.page_content
            metadata = doc_obj.metadata
            prompt += doc
        
                
        # 创建请求体
        body = {
            "prompt": prompt,
            "max_tokens": 512
        }
        
        # 发送 POST 请求
        response = requests.post(url, headers=headers, data=json.dumps(body))

        # 检查响应的状态码
        if response.status_code == 200:
            # 将响应的内容解析为 JSON
            result = response.json()

            # Extract the text from the response
            answer = result['choices'][0]['text']
            
            # 提取生成的选项
            choices = result['choices']

            # 从choices中提取第一个选项的各个属性
            first_choice = choices[0] # 这是一个包含多个字典的列表。每个字典都代表一个模型生成的选项。每个选择字典都有以下的键值对：
            finish_reason = first_choice['finish_reason'] # 表示生成结束的原因
            index = first_choice['index'] # 生成文本的索引
            logprobs = first_choice['logprobs'] # 这可能是生成文本的概率分数（在此例中为None）
            text = first_choice['text'] # 大模型推理结果

            # 提取其他属性
            created = result['created'] # 时间戳，表示何时创建了这条记录
            id_value = result['id'] # 此生成请求的唯一标识符
            model = result['model'] # 用于生成文本的模型名称
            object_value = result['object'] # 表示此数据结构的类型
            truncated = result['truncated'] # 布尔值，表示生成的文本是否被截断

            # 提取usage的属性
            usage = result['usage'] # 一个字典，包含了如下三种关于此生成请求的一些统计数据
            completion_tokens = usage['completion_tokens'] # 生成的文本使用的token计数
            prompt_tokens = usage['prompt_tokens'] # prompt使用的token计数
            total_tokens = usage['total_tokens'] # 总token计数

            # Print the result
            print("\n\n> Question:")
            print(query)
            print("\n> Answer:")
            print(answer)

            # if show_sources:  # this is a flag that you can set to disable showing answers.
            #     # # Print the relevant sources used for the answer
            #     print("----------------------------------SOURCE DOCUMENTS---------------------------")
            #     for document in docs:
            #         print("\n> " + document.metadata["source"] + ":")
            #         print(document.page_content)
            #     print("----------------------------------SOURCE DOCUMENTS---------------------------")
        else:
            print("Error: ", response.status_code)
            break





if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
