import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

class Lawyer:
    _gpu_index = 0
    _num_gpus = torch.cuda.device_count()

    def __init__(self, model_name, judgment_criteria):
        # Check if a GPU is available and move the model to the GPU
        self.model_name = model_name
        self.device = torch.device(f"cuda:{Lawyer._gpu_index}" if torch.cuda.is_available() else "cpu")
        Lawyer._gpu_index = (Lawyer._gpu_index + 1) % Lawyer._num_gpus

        # Load the tokenizer and model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(self.device)
        self.eos_token_id = self.tokenizer.eos_token_id
        # Set the pad token to be the same as the EOS token if not already set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_new_tokens = 2000
        self.judgment_criteria = judgment_criteria

    def judge_contract(self, contract):
        # Prepare the input text from the prompt
        input_text = "\nContract: " + contract + self.judgment_criteria + " Answer:"

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)

        # Generate a response with custom parameters
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=10,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return response

    def improve_contract(self, contract, improvement_criteria):
        # Combine the contract and instruct prompt
        input_text = improvement_criteria + "\nContract: " + contract + "\Improved contract: "

        # Tokenize the combined input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)

        # Generate a response with custom parameters
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )

        # Decode the response
        improved_contract = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return improved_contract

    def generate_new_contract(self, prompt):
        # Prepare the input text from the prompt
        input_text = prompt + " Contract:"

        # Tokenize the input text
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True).to(self.device)

        # Start timing the generation process
        start_time = time.time()

        # Generate a response with custom parameters
        output_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=self.max_new_tokens,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.eos_token_id,
        )

        # End timing the generation process
        end_time = time.time()
        
        # Calculate the elapsed time
        elapsed_time = end_time - start_time

        # Calculate the number of tokens generated
        num_tokens_generated = output_ids.shape[1] - inputs["input_ids"].shape[1]

        # Calculate tokens per second
        tokens_per_second = num_tokens_generated / elapsed_time

        # Decode the response
        new_contract = self.tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        # Print the tokens per second for debugging purposes
        print(f"Tokens: {num_tokens_generated:.2f}")
        print(f"Seconds: {elapsed_time:.2f}")
        print(f"Tokens per second: {tokens_per_second:.2f}")

        return new_contract