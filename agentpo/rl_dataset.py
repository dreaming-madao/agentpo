import re
import logging
import datasets
import random
import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils.model import compute_position_id_with_mask

logger = logging.getLogger(__name__)

prompt_system_backup="Provide hint to solve the below problem."

verification_system_prompt_backup = """You are a rigorous verifier. Judge a solution correct only if every step is fully justified. Solutions with flawed logic, guesses, or gaps are invalid.

Core Rules:
- Verify, do not fix: Do not correct errors or fill gaps.
- Check step-by-step: In the Verification Log, justify each step. Quote it first.

Output Format:
**1. Summary** (First section)  
- **Final Verdict**: One sentence.  
  Examples:  
  - "The solution is correct."  
  - "Invalid due to a Critical Error."  
  - "Viable but has Justification Gaps."  

- **List of Findings** (Bulleted):  
  For each issue:  
  - Location: Quote the key phrase/equation.  
  - Issue: Brief description + type (**Critical Error** or **Justification Gap**).

**2. Detailed Verification Log**  
Step-by-step analysis. For each:  
- Quote the step.  
- Evaluate with justification.

---
*Example Summary:*  
**Final Verdict:** The solution is invalid due to a Critical Error.  
**List of Findings:**  
- **Location:** "From $A > B$ and $C > D$, $A-C > B-D$"  
  - **Issue:** Critical Error, Invalid inequality subtraction.  
- **Location:** "Interchanging limit and integral..."  
  - **Issue:** Justification Gap, No justification for exchange.
"""

prompt_system="Rewrite the question below to make it easier to understand."

verification_system_prompt ="""Given a question and its current solution, analyze the solution and provide concise, specific feedback identifying any errors, logical gaps, or missing justifications. 
Do not rewrite the solution, only highlight issues. 
"""

class RLHFCustomDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            if self.config["dataset_num"]!=-1 and "test" not in parquet_file:
                dataframe = dataframe.select(range(self.config["dataset_num"]))
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            self.dataframe = self.dataframe.filter(
                lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(self.dataframe)}")

    def _build_messages(self, example: dict):
        query=example[self.prompt_key]
        if isinstance(query,list):
             messages=query
        else:
            if self.config["cooperation_mode"]=='critic':
                # solution=random.choice(example["solutions"])
                solution=example["solutions"][0]
                example['solutions']=solution
                query=f"Question: {query}\n\nCurrent solution: {solution}"
                system_prompt=verification_system_prompt 
            
            elif self.config["cooperation_mode"]=='assistant':
                system_prompt=prompt_system

            messages=[
                {'role':'system','content':system_prompt},
                {'role':'user','content':query}
            ]

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                for segment in re.split("(<image>|<video>)", content):
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages


    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            multi_modal_data = {}

            images = None
            if self.image_key in row_dict:
                images = [process_image(image) for image in row_dict.pop(self.image_key)]
                multi_modal_data["image"] = images

            videos = None
            if self.video_key in row_dict:
                videos = [process_video(video) for video in row_dict.pop(self.video_key)]
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data
            row_dict["multi_modal_inputs"] = dict(model_inputs)

            # second_per_grid_ts isn't used for training, just for mrope
            row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages
        
        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt # array of strings

        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict
    
if __name__ == "__main__":
    from verl.utils import hf_processor, hf_tokenizer
    local_path="/home/sunl/verl_rl/ckpts/Qwen_Qwen2.5-Math-7B"

    train_files=['data/math8k/math8k_hard_solutions_1000.parquet']
    test_files=['data/math8k/test_solutions.parquet']

    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)
   
    config={
        "train_files":train_files,
        "val_files":test_files,
        "train_batch_size":4,
        "max_prompt_length":256,
        "max_response_length":1500,
        "filter_overlong_prompts":True,
        "prompt_key":"problem",
        "truncation":'left',
        "shuffle": True,
        "custom_cls": None,
        "dataset_num":20,
        "cooperation_mode":"critic" # "critic","assistant"
    }
  
    train_dataset = RLHFCustomDataset(
        data_files=test_files,
        tokenizer=tokenizer,
        processor=processor,
        config=config,
    )
    data=next(iter(train_dataset))

    print('data')