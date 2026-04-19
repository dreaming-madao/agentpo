import random
import os
import argparse
import time
from vllm import LLM, SamplingParams
from datetime import datetime
from tqdm import tqdm


import torch
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

# available_gpus = "6"
# os.environ["CUDA_VISIBLE_DEVICES"] = available_gpus

SYSTEM_base = "You are a helpful AI assistant."

SYSTEM = "Rewrite the question below to make it easier to understand."

verification_system_prompt ="""Given a question and its current solution, analyze the solution and provide concise, specific feedback identifying any errors, logical gaps, or missing justifications. 
Do not rewrite the solution, only highlight issues. 
"""

openai_api_key_ai_lab = ''

llm_model_dict = {
    "Qwen2.5-72b": "http://174.34.106.21:10001/v1",
    # "Qwen2.5-7b": "http://174.34.106.21:10003/v1",
    "Qwen2.5-14b": "http://174.34.106.20:10001/v1",
    # "Qwen2.5-7b": "http://174.34.106.20:10001/v1",
    "Qwen3-4b": "http://174.34.106.20:10003/v1",
    "llm-grpo": "http://174.34.106.20:10060/v1",
    # "Llama-3.1-8B": "http://174.34.106.20:10003/v1",
    "Llama-3.2-3B": "http://174.34.106.21:10005/v1",
    "Llama-3.1-8B": "http://174.34.106.21:10005/v1",
    "aliyuncs-api": "https://vpc-cn-beijing.dashscope.aliyuncs.com/compatible-mode/v1",
}

configs = {
    "Qwen-max-uaes-1206": ["qwen-max", openai_api_key_ai_lab],
    "Qwen-plus-uaes-1206": ["qwen-plus", openai_api_key_ai_lab],
    "deepseek-r1-uaes-1206": ["deepseek-r1", openai_api_key_ai_lab],
    "deepseek-v3-uaes-1206": ["deepseek-v3", openai_api_key_ai_lab],
    "qwq-32b-preview-uaes-1206": ["qwq-32b-preview", openai_api_key_ai_lab],
}

uaes_model_list = ["Qwen2.5-72b", "Qwen2.5-14b", "Qwen2.5-7b", "Qwen2.5-3b","Llama-3.2-3B", "Llama-3.1-8B", "Qwen3-4b"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="aime24,math500,olympiadbench,minerva_math,amc23", type=str)
    parser.add_argument("--data_dir", default="/home/sunl/verl_rl/evaluate_math/evaluation/data", type=str)
    parser.add_argument("--model_name_or_path", default="/home/sunl/verl_rl/ckpts/Qwen2.5-3B-Instruct", type=str)
    parser.add_argument("--output_dir", default="runs/hard1000_dapo_last", type=str)
    parser.add_argument("--prompt_type", default="llama3", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=5, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens_per_call", default=768, type=int)
    parser.add_argument("--shuffle", default=False)
    parser.add_argument("--use_vllm", default=True)
    parser.add_argument("--save_outputs", default=True)
    parser.add_argument("--overwrite",  default=True)
    parser.add_argument("--use_safetensors", default=False)
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--use_pass_k", type=bool, default=True)
    parser.add_argument("--actor_model", type=str, default="Qwen-plus-uaes-1206")
    parser.add_argument("--cooperation_mode", type=str, default="critic")
    parser.add_argument("--apply_chat_template",default=True, help="Apply chat template to prompt.",)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )
    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def get_solution(prompt_lst, llm_model, cooperation_mode):
    BATCH_SIZE=64
    
    if llm_model in uaes_model_list:
        llm_model_base = llm_model
        openai_api_base = llm_model_dict[llm_model_base]
        openai_api_key = "empty"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=6000)
        model = client.models.list().data[0].id

    else:
        llm_model_base = "aliyuncs-api"
        openai_api_base = llm_model_dict[llm_model_base]
        model, openai_api_key = configs[llm_model]
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base, timeout=6000)

    if cooperation_mode=="critic":
        system_prompt='You have an opportunity to improve your solution. Please review Current Solution and Comment carefully. Correct errors and fill justification gaps if any.'
        
    elif cooperation_mode in ["base", "assistant"]:
        system_prompt='Please reason step by step, and put your final answer within \\boxed{{}}.'

    all_responses = []

    for i in range(0, len(prompt_lst), BATCH_SIZE):
        batch_prompts = prompt_lst[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}, prompts {i} to {min(i + BATCH_SIZE - 1, len(prompt_lst) - 1)}") 
        
        batch_messages = []
        for prompt in batch_prompts:
            prompt += " Let's think step by step and output the final answer within \\boxed{{}}."
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            batch_messages.append(messages)

        batch_completions = []
        for messages in batch_messages:
            completion = client.chat.completions.create(
                messages=messages,
                model=model,
                stream=True,
                max_tokens=2048,
                temperature=0,
                top_p=1.0
            )
            batch_completions.append(completion)

        batch_responses = []
        for completion in batch_completions:
            response_item = ""
            try:
                for chunk in completion: 
                    content = chunk.choices[0].delta.content
                    if content is not None:
                        response_item += content
            except Exception as e:
                print(f"Error processing a completion in batch: {e}")
                response_item = "" 
            batch_responses.append(response_item)

        all_responses.extend(batch_responses)

    return all_responses


def prepare_data(data_name, args):
    if args.cooperation_mode=="critic":
        is_org_data=False
        data_dir=f"/home/sunl/verl_rl/evaluate_math/outputs/runs/Base_{args.actor_model}"
        
    elif args.cooperation_mode in ["base","assistant"]:
        is_org_data=True
        data_dir=args.data_dir

    examples = load_data(data_name, args.split, data_dir, is_org_data)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start: len(examples) if args.end == -1 else args.end]

    # get out_file name
    dt_string = datetime.now().strftime("%m-%d_%H-%M")
    model_name = "/".join(args.model_name_or_path.split("/")[-2:])
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []
    if not args.overwrite:
        processed_files = [
            f
            for f in os.listdir(f"{output_dir}/{data_name}/")
            if f.endswith(".jsonl") and f.startswith(out_file_prefix)
        ]
        for f in processed_files:
            processed_samples.extend(
                list(load_jsonl(f"{output_dir}/{data_name}/{f}"))
            )

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples, out_file

def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if args.use_vllm:
        llm = LLM(
            model=args.model_name_or_path,
            tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            trust_remote_code=True,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )

        # tokenizer = AutoTokenizer.from_pretrained(
        #         args.model_name_or_path, trust_remote_code=True
        #     )
        tokenizer = None
        if args.apply_chat_template:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, trust_remote_code=True
            )
    else:
        llm, tokenizer = load_hf_lm_and_tokenizer(
            model_name_or_path=args.model_name_or_path,
            load_in_half=True,
            use_fast_tokenizer=True,
            use_safetensors=args.use_safetensors,
        )

    # infer & eval
    data_list = args.data_names.split(",")
    results,cost_times = [],[]
    for data_name in data_list:
        start_time=time.time()
        results.append(main(llm, tokenizer, data_name, args))
        cost_times.append(time.time()-start_time)

    print("=====================================")
    for data_name, result in zip(data_list,results):
        print(f"\n\n{data_name}: {result}")
    print("=====================================")

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )
    cost_times.append(sum(cost_times)/len(cost_times))

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))
    print("\t".join([f"{cost_time:.1f}".ljust(pad, " ") for cost_time in cost_times]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(llm, tokenizer, data_name, args):
    examples, processed_samples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, ",remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for example in tqdm(examples, total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans
        full_prompt = construct_prompt(example, data_name, args)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "code"
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    if args.cooperation_mode == "base":
        system_prompt=SYSTEM_base
        input_prompts = [
            sample["question"] for sample in samples for _ in range(args.n_sampling)
        ]
    
    elif args.cooperation_mode =="assistant":
        system_prompt=SYSTEM
        input_prompts = [
            sample["question"] for sample in samples for _ in range(args.n_sampling)
        ]
        
    elif args.cooperation_mode=="critic":
        system_prompt=verification_system_prompt
        input_prompts=[]
        for sample in samples:
            ques=sample["question"]
            sol=sample["code"][0]
            for _ in range(args.n_sampling):
                input_prompts.append(f"Question: {ques}\n\nCurrent Solution: {sol}")
        
    if args.apply_chat_template:
        input_prompts = [(prompt,
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt}, # qwen 自带了
                    {"role": "user", "content": prompt.strip()},
                ],
                tokenize=False,
                add_generation_prompt=True,
            ))
            for prompt in input_prompts
        ]

    print(input_prompts[0][1])

    remain_prompts = input_prompts
    remain_prompts = [(i, question, prompt) for i, (question,prompt) in enumerate(remain_prompts)]
    end_prompts = []
    mid_results=[]
    
    max_func_call = 1 if args.prompt_type in ["cot", "pal"] else 4

    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]

    if args.prompt_type in ["cot"]:
        stop_words.append("\n\nQuestion:")
    if args.prompt_type in ["pal", "tool-integrated", "jiuzhang_tora"]:
        stop_words.extend(["\n\n---", "```output"])
    elif args.prompt_type in ["wizard_zs", "platypus_fs"]:
        stop_words.extend(["Instruction", "Response"])
    elif "jiuzhang" in args.prompt_type:
        stop_words.append("\n\n## Question")
    elif "numina" in args.prompt_type:
        stop_words.append("\n### Problem")
    elif "pure" in args.prompt_type:
        stop_words.append("\n\n\n")

    # start inference
    # measure time use
    start_time = time.time()
    for epoch in range(max_func_call):
        print("-" * 20, "Epoch", epoch)
        current_prompts = remain_prompts
        if len(current_prompts) == 0:
            break

        remain_prompts = []
        remain_codes = []

        if args.cooperation_mode in ["assistant","critic"]:
            # get all outputs
            prompts = [item[2] for item in current_prompts]
            if args.use_vllm:
                outputs = llm.generate(
                    prompts,
                    SamplingParams(
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        max_tokens=args.max_tokens_per_call,
                        n=1,  # n: Number of output sequences to return for the given prompt.
                        stop=stop_words,
                        stop_token_ids=(
                            [151645, 151643]
                            if "qwen2" in args.model_name_or_path.lower()
                            else None
                        ),
                    ),
                )

                outputs = sorted(
                    outputs, key=lambda x: int(x.request_id)
                )  # sort outputs by request_id
                outputs = [output.outputs[0].text for output in outputs]
                # token_lens = [len(tokenizer.encode(output)) for output in outputs]  # 计算 token 长度
            else:
                outputs = generate_completions(
                    model=llm,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    max_new_tokens=args.max_tokens_per_call,
                    batch_size=16,
                    stop_id_sequences=stop_words,
                )

            assert len(outputs) == len(current_prompts)

            # process all outputs
            prompt_lst=[]
            for current_prompt,output in zip(current_prompts, outputs):
                output = output.rstrip()
                question=current_prompt[1]
                if args.cooperation_mode=="assistant":
                    prompt=f"Problem: {question} (Hint: {output})"
                
                elif args.cooperation_mode=="critic":
                    prompt=f"{question}\n\nComment: {output}"

                prompt_lst.append(prompt)
                if epoch==0:
                    mid_results.append(output)

            if epoch==0:
                print(prompt_lst[0])
        else:
            prompt_lst = [current_prompt[1] for current_prompt in current_prompts]
        
        outputs=get_solution(prompt_lst, args.actor_model, args.cooperation_mode)

        for (i, question, query), output in zip(current_prompts, outputs):
            output = output.rstrip()
            query += output
            if args.prompt_type == "pal":
                remain_prompts.append((i, question, query))
                if "```python" in output:
                    output = extract_program(query)
                remain_codes.append(output)
            elif "boxed" not in output and output.endswith("```"):
                program = extract_program(query)
                remain_prompts.append((i, question, query))
                remain_codes.append(program)
            elif args.prompt_type == "cot":
                end_prompts.append((i, query))
            else:
                end_prompts.append((i, query))

        # execute the remain prompts
        remain_results = executor.batch_apply(remain_codes)
        for k in range(len(remain_prompts)):
            i, question, query = remain_prompts[k]
            res, report = remain_results[k]
            exec_result = res if res else report
            if "pal" in args.prompt_type:
                exec_result = "\\boxed{" + exec_result + "}"
            exec_result = f"\n```output\n{exec_result}\n```\n"
            query += exec_result
            # not end
            if epoch == max_func_call - 1:
                query += "\nReach max function call limit."
            remain_prompts[k] = (i, question, query)

    # unsolved samples
    print("Unsolved samples:", len(remain_prompts))
    end_prompts.extend(remain_prompts)
    # sort by idx
    end_prompts = sorted(end_prompts, key=lambda x: x[0])

    # remove input_prompt from end_prompt
    codes = []
    assert len(input_prompts) == len(end_prompts)
    for i in range(len(input_prompts)):
        _, end_prompt = end_prompts[i]
        code = end_prompt.split(input_prompts[i][1])[-1].strip()
        for stop_word in stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    # extract preds
    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]
    time_use = time.time() - start_time

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        
        mid_result=mid_results[i] if args.cooperation_mode in ["assistant","critic"] else ""
        
        code = codes[i * args.n_sampling: (i + 1) * args.n_sampling]
        result = results[i * args.n_sampling: (i + 1) * args.n_sampling]
        preds = [item[0] for item in result]
        reports = [item[1] for item in result]
        # token_lengths = token_lens[i * args.n_sampling : (i + 1) * args.n_sampling]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )

        sample.pop("prompt")
        # sample.update({"code": code, "pred": preds, "report": reports, "token_length": token_lengths})
        sample.update({"code": code, "pred": preds, "report": reports,"mid_result":mid_result})
        all_samples.append(sample)

    # add processed samples
    if args.use_pass_k:
        print('✨pass_k will be used!')

    all_samples.extend(processed_samples)
    all_samples, result_json, score_mat = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
        use_pass_k=args.use_pass_k
    )

    # save outputs
    if len(processed_samples) < len(all_samples) and args.save_outputs:
        save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    score_mat_file = out_file.replace(".jsonl", f"_score_mat.npy")
    np.save(score_mat_file, np.array(score_mat))

    with open(
            # out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
            out_file.replace(".jsonl", f"_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)

    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
