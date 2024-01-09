# predict_model.py
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
import pandas as pd

def generate_summaries(texts, dataloader):
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    summaries = []
    for batch in dataloader:
        texts = batch['input_text']
        for text in texts:
            summary = generate_summary(text, model, tokenizer)
            summaries.append(summary)

    return summaries


def generate_summary(text, fine_tuned_model_path, ):
    fine_tuned_model = BartForConditionalGeneration.from_pretrained(fine_tuned_model_path)
    tokenizer = BartTokenizer.from_pretrained(fine_tuned_model_path)
    
    summarizer = pipeline("summarization", model=fine_tuned_model, tokenizer=tokenizer)
    summary = summarizer(input_text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return summary[0]['summary_text']


def main():
    # List of texts for summarization
    texts = [
        "Text 1 for summarization...",
        "Text 2 for summarization...",
        # Add more texts as needed
    ]

    # Fine-tuned model path
    fine_tuned_model_path = "fine_tuned_summarizer"

    # Generate summaries for all texts
    summaries = generate_summary(texts, fine_tuned_model_path)

    # Create a DataFrame
    df = pd.DataFrame({'Original Text': texts, 'Generated Summary': summaries})

    # Display the DataFrame
    print(df)

if __name__ == "__main__":
    main()
    
    
    
    