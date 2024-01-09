from transformers import BartTokenizer, BartForConditionalGeneration

class ScientificPaperSummarizer:
    def __init__(self, model_name="facebook/bart-large-cnn"):
        """ Fine tuned BART model for Paper Summarization
    
            Args:
            model_name: checkpoint
                
        """
        # Instantiate and load model weights for Bart model, allowing us to access configuration parameters (number of layers etc.)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        
        # BartTokenizer loads tokenizer for specific Bart model
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
    
