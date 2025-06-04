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
            'agriculture': """As an expert agricultural AI assistant, analyze this farm image and provide detailed insights.
Focus your analysis on:
- Plant health assessment and disease identification
- Crop growth stages and development conditions  
- Pest, weed, and pathogen detection
- Soil health indicators if visible
- Irrigation and water management needs
- Nutrient deficiency symptoms
- Harvest readiness evaluation
- Environmental stress factors
- Recommended agricultural interventions

Question: {question}

Provide practical, actionable advice that farmers can implement immediately. Be specific about treatments, timing, and methods.""",

            'disease_diagnosis': """You are an agricultural pathologist. Examine this image for plant diseases and disorders.

Analyze for:
- Fungal, bacterial, and viral diseases
- Nutrient deficiencies and toxicities
- Environmental stress symptoms
- Pest damage patterns
- Disease progression stages
- Treatment recommendations
- Prevention strategies

Question: {question} """,
            
            'general': "Answer the following agricultural question about this farm image: {question}"
        }
        logger.info("VLMProcessor initialized with agricultural focus: %s", settings.VLM_MODEL_ID)
    
    def generate_response(self, query_type, image_key, **kwargs):
        try:
            # Default to agriculture template if not specified
            if query_type not in self.prompt_templates:
                query_type = 'agriculture'
                
            prompt_template = self.prompt_templates.get(query_type, self.prompt_templates['general'])
            
            # Format the prompt with the question
            prompt = prompt_template.format(question=kwargs.get('question', ''))
            
            logger.debug(f"Invoking VLM with agricultural prompt for query type: {query_type}")
            logger.debug(f"Using image key: {image_key}")
            
            response = self.aws_client.invoke_vlm(prompt, image_key)
            if not response:
                logger.error("No response from VLM")
                return "Unable to analyze the agricultural image. Please try again."
            
            logger.debug(f"Raw VLM response: {response}")
            
            try:
                response_json = json.loads(response)
                return response_json.get('content', [{}])[0].get('text', "No agricultural analysis generated.")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse VLM response: {response}")
                return response
                
        except Exception as e:
            logger.error(f"VLM processing error: {str(e)}")
            return "Error occurred during agricultural image analysis."
