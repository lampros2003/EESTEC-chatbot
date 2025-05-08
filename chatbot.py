from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


class ChatBot:
    def __init__(self, model_name="Qwen/Qwen2-1.5B-Instruct"):

        # Qwen/Qwen2-3B-Instruct einai poli megalo
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",  # Automatically places layers on GPU(s)/CPU
            torch_dtype=torch.float16,  # Use half‐precision for speed & memory
        )
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def retrieve_context(self) -> str:

        return """
        quantum computing is a field of study that focuses on the application of quantum mechanics to the field of computing. It explores how quantum phenomena can be harnessed to perform computations that are infeasible for classical computers. Quantum computing has the potential to revolutionize fields such as cryptography, optimization, and drug discovery by enabling faster and more efficient algorithms.
        Quantum computers use quantum bits (qubits) instead of classical bits, allowing them to represent and process information in fundamentally different ways. This enables quantum computers to solve certain problems exponentially faster than classical computers. However, building practical quantum computers is still a significant challenge due to issues such as qubit coherence, error rates, and scalability.
        """

    def build_prompt(self, context: str, question: str) -> str:

        return (
            "Answer the following question based only on the context provided.\n\n"
            f"Context:\n{context.strip()}\n\n"
            f"Question:\n{question.strip()}\n\n"
            "Answer in one very small sentence with less than 20 words.\n"
            "OUTPUT **ONLY** THE ANSWER—do _not_ include any extra explanation or quotes."
        )

    def generate_answer(self, question: str) -> str:

        context = self.retrieve_context()
        prompt = self.build_prompt(context, question)
        chat_prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(chat_prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.2,
            top_p=1.0,
            num_beams=1,
            top_k=0,
            repetition_penalty=1.0,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        raw = self.tokenizer.decode(out[0], skip_special_tokens=True)

        # Strip away any echoed prompt
        answer = raw.split(
            "OUTPUT **ONLY** THE ANSWER—do _not_ include any extra explanation or quotes."
        )[-1].strip()
        return answer

    def chat(self):

        print("Welcome to Qwen2-Instruct local chatbot. Type ‘exit’ or Ctrl-C to quit.")
        while True:
            try:
                user_q = input("\nYou: ").strip()
                if user_q.lower() in ("exit", "quit"):
                    break
                answer = self.generate_answer(user_q)
                print(
                    f"Bot: {answer[9:] if answer.startswith('assistant') else answer}"
                )
            except KeyboardInterrupt:
                print("\nExiting.")
                break


if __name__ == "__main__":
    chatbot = ChatBot()
    chatbot.chat()
