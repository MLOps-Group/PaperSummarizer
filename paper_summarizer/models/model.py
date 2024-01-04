import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import load_dataset

class ScientificPaperSummarizer:
    """ Basic neural network class. 
    
    Args:
        in_features: number of input features
        out_features: number of output features
    
    """
    def __init__(self, model_name="facebook/bart-large-cnn"):
        # BartTokenizer loads tokenizer for specific Bart model
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        # Instantiate and load model weights for Bart model, allowing us to access configuration parameters (number of layers etc.)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.dataset = load_dataset("scientific_papers")

        # Converts input to format that can be pre-processed
        self.tokenized_dataset = self.tokenize_dataset()
        
    def tokenizer():
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: input tensor expected to be of shape [N,in_features]

        Returns:
            Output tensor with shape [N,out_features]

        """
        return self.l2(self.r(self.l1(x)))