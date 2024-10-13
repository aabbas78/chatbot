import gradio as gr
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments
import PyPDF2
from datasets import Dataset

# Load the pre-trained model and tokenizer
model_name = "distilgpt2"  # You can also use "gpt2" or another model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Global variable to hold the model
model = None

# Function to extract text from uploaded PDF file
def extract_text_from_uploaded_pdf(pdf_file):
    text = ""
    try:
        with open(pdf_file, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in range(len(reader.pages)):
                text += reader.pages[page].extract_text()
        print("PDF extraction complete. Text length:", len(text))
    except Exception as e:
        print("Error reading PDF:", str(e))
    return text

# Fine-tune the model based on the uploaded PDF
def fine_tune_model(pdf_file):
    global model
    print("Starting model fine-tuning...")
    
    # Extract text from the PDF
    pdf_text = extract_text_from_uploaded_pdf(pdf_file)
    
    if not pdf_text.strip():  # Check if text extraction was successful
        print("Error: No text found in the PDF file.")
        return "Error: No text found in the PDF file."

    # Prepare dataset for fine-tuning
    data = {"text": [pdf_text]}
    dataset = Dataset.from_dict(data)

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./fine_tuned_aircraft_model",  # Directory to save the model
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust based on your dataset size
        per_device_train_batch_size=1,  # Use 1 for debugging
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # Load the base model for fine-tuning
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Fine-tune the model using Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    try:
        # Start training
        trainer.train()
        print("Model fine-tuning complete.")
    except Exception as e:
        print("Error during training:", str(e))
        return f"Error during training: {str(e)}"

    # Save the fine-tuned model and tokenizer
    model.save_pretrained("fine_tuned_aircraft_model")
    tokenizer.save_pretrained("fine_tuned_aircraft_model")

    return "Model fine-tuned successfully!"

# Create a pipeline for text generation (chatbot)
def create_chatbot_pipeline():
    global model
    if model is None:
        model = GPT2LMHeadModel.from_pretrained("fine_tuned_aircraft_model")
    return pipeline("text-generation", model=model, tokenizer=tokenizer)

# Chatbot function that answers questions based on the fine-tuned model
def chatbot(input_text):
    global model
    if model is None:
        print("Error: Model is not loaded.")
        return "Error: Model is not loaded."
    
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    
    try:
        # Generate the response using the input text
        response = generator(input_text, max_length=100, num_return_sequences=1)[0]["generated_text"]
        print("Response generated successfully.")
        return response.strip()  # Strip any leading/trailing whitespace
    except Exception as e:
        print("Error generating response:", str(e))
        return f"Error generating response: {str(e)}"

# Create a Gradio interface
def main_interface(pdf_file, input_text):
    fine_tune_message = fine_tune_model(pdf_file)  # Fine-tune the model
    if "Error" in fine_tune_message:
        return fine_tune_message  # Return error message if any during fine-tuning

    create_chatbot_pipeline()  # Create the chatbot pipeline
    return chatbot(input_text)  # Generate a response to the input question

# Gradio interface inputs and outputs
interface = gr.Interface(
    fn=main_interface, 
    inputs=[gr.File(label="Upload PDF"), gr.Textbox(label="Ask a Question")], 
    outputs="text", 
    title="Aircraft Maintenance Chatbot"
)

# Launch the interface
if __name__ == "__main__":
    interface.launch()
