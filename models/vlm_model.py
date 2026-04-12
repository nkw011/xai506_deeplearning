"""
Qwen2-VL wrapper for Visual Question Answering (VQA).
Model: Qwen/Qwen2-VL-7B-Instruct

Uses flash_attention_2 for faster inference (flash-attn is pre-installed).
Supports multi-turn conversation on the same image.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class Qwen2VLQA:
    MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"

    def __init__(self):
        self.model = None
        self.processor = None

    def load(self):
        print(f"[Qwen2-VL] Loading {self.MODEL_ID} ...")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="auto",
        )
        self.model.eval()
        print("[Qwen2-VL] Model loaded.")

    def _generate(self, messages: list, max_new_tokens: int = 256) -> str:
        """Internal: apply chat template, generate, decode."""
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=[msg["content"][0]["image"] for msg in messages if isinstance(msg["content"], list) and msg["content"][0].get("type") == "image"],
            return_tensors="pt",
        )
        # Move to the same device as the model's first parameter
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        new_ids = generated_ids[:, input_len:]
        return self.processor.batch_decode(new_ids, skip_special_tokens=True)[0].strip()

    def answer(self, image: Image.Image, question: str) -> str:
        """Answer a single question about an image.

        Args:
            image: PIL Image (RGB).
            question: Natural language question string.

        Returns:
            Model's answer as a string.
        """
        assert self.model is not None, "Call load() first."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]
        return self._generate(messages)

    def answer_multiturn(
        self,
        image: Image.Image,
        history: list[dict],
        question: str,
    ) -> tuple[str, list[dict]]:
        """Multi-turn VQA: continue a conversation about the same image.

        Args:
            image: PIL Image (RGB). Used only for the first turn.
            history: List of previous {"role": ..., "content": ...} messages.
                     Pass [] for the first question.
            question: New question to ask.

        Returns:
            answer: Model's response string.
            updated_history: New history with the latest exchange appended.
        """
        assert self.model is not None, "Call load() first."

        if not history:
            # First turn – attach the image
            new_user_msg = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        else:
            # Subsequent turns – text only
            new_user_msg = {
                "role": "user",
                "content": [{"type": "text", "text": question}],
            }

        messages = history + [new_user_msg]
        answer = self._generate(messages)

        updated_history = messages + [
            {"role": "assistant", "content": [{"type": "text", "text": answer}]}
        ]
        return answer, updated_history

    def caption(self, image: Image.Image) -> str:
        """Generate a detailed description of the image without a question.

        Args:
            image: PIL Image (RGB).

        Returns:
            Description string.
        """
        return self.answer(image, "Describe this image in detail.")
