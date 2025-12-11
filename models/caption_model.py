import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image

class ImageCaptioningModel:
    def __init__(self, mode="fast"):
        model_name = "microsoft/Florence-2-base"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Data type compatible for low ended CPU. as i don't have a GPU.
        self.dtype = torch.bfloat16 if self.device == "cpu" else torch.float16

        self.mode = mode

        print(f"Loading Florence-2 in {self.mode.upper()} mode...")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True
        ).to(self.device)

        # Enable PyTorch 2.0 compile for more speed
        self.model = torch.compile(self.model)  

        self.processor = AutoProcessor.from_pretrained(
            model_name,
            trust_remote_code=True
        )

    def generate_caption(self, image: Image.Image):
        prompt = "<CAPTION>"

        # Optimising the model to work FAST. as we are using low configuratin settings.
        if self.mode == "fast":
            max_tokens = 60
            image = image.resize((448, 448))  # smaller image (Choti Image)
        else:
            max_tokens = 128  # HQ mode (Achhi Quality)

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.device, self.dtype)

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_tokens,
                do_sample=False
            )

        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task="<CAPTION>"
        )

        return parsed["<CAPTION>"]
