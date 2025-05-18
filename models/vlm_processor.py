import json
from config.settings import settings
from services.aws_client import AWSClient
from utils.logger import logger

class VLMProcessor:
    def __init__(self):
        self.aws_client = AWSClient()
        self.prompt_templates = {
            'describe': "Describe the scene in this image in detail, focusing on objects, their positions, and any notable characteristics.",
            'count': "Count the number of {object}s visible in this image. Only provide the number.",
            'identify': "Identify and list all the {category} visible in this image.",
            'general': "Answer the following question about this image: {question}"
        }
        logger.info("VLMProcessor initialized with model: %s", settings.VLM_MODEL_ID)
    
    def generate_response(self, query_type, image_key, **kwargs):
        try:
            prompt_template = self.prompt_templates.get(query_type, self.prompt_templates['general'])
            
            if query_type == 'general':
                prompt = prompt_template.format(question=kwargs.get('question', ''))
            else:
                prompt = prompt_template.format(**kwargs)
            
            logger.debug(f"Invoking VLM with prompt: {prompt}")
            logger.debug(f"Using image key: {image_key}")
            
            response = self.aws_client.invoke_vlm(prompt, image_key)
            if not response:
                logger.error("No response from VLM")
                return "I couldn't process that request. Please try again."
            
            logger.debug(f"Raw VLM response: {response}")
            
            try:
                response_json = json.loads(response)
                return response_json.get('content', [{}])[0].get('text', "No response generated.")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse VLM response: {response}")
                return response
                
        except Exception as e:
            logger.error(f"VLM processing error: {str(e)}")
            return "I encountered an error processing your request."