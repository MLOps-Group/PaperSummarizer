from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import pandas as pd

def test_model():
    texts = [
        "Text 1 for summarization...",
        "Text 2 for summarization...",
        # Add more texts as needed
    ]

    # Fine-tuned model path
    #fine_tuned_model_path = "fine_tuned_summarizer"

    # Generate summaries for all texts
    #fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path)
    #tokenizer = BartTokenizer.from_pretrained(fine_tuned_model_path)
    
    #summarizer = pipeline("summarization", model=fine_tuned_model, tokenizer=tokenizer)
    #summary = summarizer(texts, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    #print(summary[0]['summary_text'])
    pass

if __name__ == "__main__":
    test_model()
    
    

